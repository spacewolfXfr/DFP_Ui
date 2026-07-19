#!/usr/bin/python3

from __future__ import annotations

import dataclasses
import typing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.axes import Axes

Colortype = tuple[float,float,float,float]

import pandas as pd

from pitot.geodesy import bearing, destination, distance

from traffic.core import Traffic,Flight,FlightIterator
from traffic.data import samples,airports,runways


from traffic.algorithms import filters
from cartopy import crs as ccrs

from .profiling.ProblemGenerator import AC_PP_Problem,ACStats
from .Dubins import Pose3D

from pyproj import Transformer

lambert_fr_transformer = Transformer.from_crs(
    "EPSG:4326",   # WGS84 (lat, lon)
    "EPSG:9794",   # Lambert-93
)

NM_TO_METERS = 1852

#################### Helpers ####################

@dataclasses.dataclass
class LatlonPose:
    latitude:float
    longitude:float
    altitude:float  # in fts
    bearing:float   # in degrees
    
    def to_pose3D(self,transformer:Transformer,to_NM:bool=False) -> Pose3D:
        x,y = transformer.transform(self.latitude,self.longitude)
        if to_NM:
            return Pose3D(x/NM_TO_METERS,y/NM_TO_METERS,self.altitude,np.deg2rad(90-self.bearing))
        else:
            return Pose3D(x,y,self.altitude,np.deg2rad(90-self.bearing))
        
    def project(self,distance:typing.Annotated[float,"m"]) -> LatlonPose:
        nlat,nlon,ndir = destination(self.latitude,self.longitude,self.bearing,distance)
        return LatlonPose(nlat,nlon,self.altitude,ndir+180)


def flight_landing(flight:Flight,candidate_airports:typing.Iterable[str]) -> typing.Optional[str]:
    for c in candidate_airports:
        landing_seq = flight.aligned_on_ils(c)
        if landing_seq.has():
            return c
    
    return None
            

def get_enddatas(flight:Flight,airport:str):
    df = flight.data.sort_values("timestamp")
    ends = flight.aligned_on_ils(airport,0.05).next()
    end_df = ends.data.sort_values("timestamp") # type: ignore
    return df.iloc[0],end_df.iloc[0]

def get_runway(airport:str,rw_name:str):
    try:
        return airports[airport].runways.data.query(f"name=='{rw_name}'").iloc[0] # type: ignore
    except (ValueError, IndexError):
        print(f"WARNING: Runway {rw_name} not found for airport {airport}")
        print(airports[airport].runways.data)
        return None

def get_other_runway_name(rw_name:str) -> str:
    """Given a runway name, try to find its name in the other direction.
    
    The rules are as follows:
    - The name in split in two parts: a number (the first two characters) and an optional letter (the third character, if it exists)
    - The number is converted to the other direction by adding 18 if it is less than 18, and substracting 18 otherwise. If the result is 0, it is replaced by 36.
    - The letter, if it exists, is converted as follows: C is unchanged, R becomes L and L becomes R.

    Args:
        rw_name (str): Runway name

    Returns:
        str: Its guessed name in the other direction
    """
    rw_num = int(rw_name[:2])
    rw_numbis = rw_num + (18 if rw_num < 18 else -18)
    if rw_numbis == 0:
        rw_numbis = 36
    if len(rw_name) == 2:
        return f"{rw_numbis:02}"
    else:
        if rw_name[2] == 'C':
            return f"{rw_numbis:02}C"
        elif rw_name[2] == 'R':
            return f"{rw_numbis:02}L"
        else:
            return f"{rw_numbis:02}R"
@dataclasses.dataclass
class FlightEndpoints:
    start:LatlonPose
    start_time:pd.Timestamp
    end:LatlonPose
    end_time:pd.Timestamp
    initial_end_time:pd.Timestamp
    dest_ICAO:str
    dest_runway:str
    stats:ACStats   # Speed is expected in NM / minute, turn radius in NM
    
    @property
    def id(self) -> int:
        return self.stats.id
    
    @id.setter
    def id(self,_id:int):
        self.stats.id = _id
    
    def straight_update(self,duration:pd.Timedelta) -> FlightEndpoints:
        if self.start_time+duration > self.end_time:
            raise ValueError("End time exceeded")
        speed = self.stats.airspeed # NM / min
        dist = (speed * duration.total_seconds()/60) * NM_TO_METERS
        nlat,nlon,ndir = destination(self.start.latitude,self.start.longitude,self.start.bearing,dist)
        new_start = LatlonPose(nlat,nlon,self.start.altitude,ndir+180)
        
        
        return FlightEndpoints(
            new_start,
            self.start_time+duration,
            self.end,
            self.end_time,
            self.initial_end_time,
            self.dest_ICAO,
            self.dest_runway,
            self.stats
        )
    
    def to_AC_PP_Problem(self,transformer:Transformer,timeshifts:list[pd.Timedelta],flatten:bool=True) -> AC_PP_Problem:
        """
        Export from this flight endpoints an Aircraft Path Planning problem  
        
        
        Args:
            transformer (Transformer): Convert latlon to a flat projection. Assumes final projection is in meters (which are then converted in NM)
            timeshifts (list[pd.Timedelta]): Allowed time differences with respect to current time of arrival (different planning objectives)
            flatten (bool, optional): Set all vertical values to 0. Defaults to True.

        Returns:
            AC_PP_Problem: _description_
        """
        
        sx,sy = transformer.transform(self.start.latitude,self.start.longitude)
        start_proj = Pose3D(sx/NM_TO_METERS,sy/NM_TO_METERS,self.start.altitude,np.deg2rad(90 - self.start.bearing))
        
        ex,ey = transformer.transform(self.end.latitude,self.end.longitude)
        end_proj = Pose3D(ex/NM_TO_METERS,ey/NM_TO_METERS,self.end.altitude,np.deg2rad(90 - self.end.bearing))
        
        if flatten:
            start_proj.z = 0.
            end_proj.z = 0.
        
        timeslots = []
        for ts in timeshifts:
            slot = ts + (self.end_time - self.start_time)
            if slot.total_seconds() >= 0:
                timeslots.append(slot.total_seconds()/60)
        
        return AC_PP_Problem(
            self.stats,
            start_proj,
            end_proj,
            timeslots=timeslots
        )
        

def flight_datapoint_to_pose(dpt:pd.Series) -> LatlonPose:
    return LatlonPose(
        dpt["latitude"],
        dpt["longitude"],
        dpt["altitude"],
        dpt["track"]
    )

def extract_flight_endpoints(flight:Flight,candidate_airports:typing.Iterable[str],
                             stats:ACStats,
                             threshold_shift:float=20) -> typing.Optional[tuple[FlightEndpoints,FlightEndpoints]]:
    """ Given a flight, a list of airports where it could have landed (ICAO codes) and a threshold shift, returns:
    - None is no landing could be confirmed
    - A pair a FlightEndpoints, one from start to first ILS confirmed alignment, the second from start to the landing runway threshold, shifted by `threshold_shift`

    Args:
        flight (Flight): Flight object
        stats (ACStats): Assumed characteristics of the aircraft
        candidate_airports (typing.Iterable[str]): List of airports where the aircraft possibly landed
        threshold_shift (float, optional): Distance to pre-shift from the landing runway threshold, in Nautical Miles. Defaults to 20 NM.

    Returns:
        typing.Optional[tuple[FlightEndpoints,FlightEndpoints]]: None if no landing could be confirmed, or a pair of endpoints, from start to either first ILS fix or shifted threshold
    """
    last_dpt    = None
    airport     = None
    runway      = None 
    for c in candidate_airports:
        landing_seq = flight.aligned_on_ils(c,0.05)
        try:
            df = next(landing_seq).data
            last_dpt = df.sort_values("timestamp").iloc[-1]
            airport = c
            runway = get_runway(airport,last_dpt["ILS"])
            break
        except StopIteration:
            continue
    
    if last_dpt is None or airport is None or runway is None:
        return None
    
    df = flight.data.sort_values("timestamp")
    first_dpt = df.iloc[0]
    
    # Turn around and convert NM to meters
    rw_proj = destination(runway.latitude, runway.longitude, runway.bearing + 180, threshold_shift * NM_TO_METERS)
    
    return FlightEndpoints(
        flight_datapoint_to_pose(first_dpt),
        first_dpt["timestamp"],
        flight_datapoint_to_pose(last_dpt),
        last_dpt["timestamp"],
        last_dpt["timestamp"],
        airport,last_dpt["ILS"],
        stats
    ), FlightEndpoints(
        flight_datapoint_to_pose(first_dpt),
        first_dpt["timestamp"],
        LatlonPose(rw_proj[0],rw_proj[1],last_dpt["altitude"],rw_proj[2]),
        last_dpt["timestamp"],
        last_dpt["timestamp"],
        airport,last_dpt["ILS"],
        stats
    )
    

    
#################### Plotting ####################

def plot(ICAO_set:typing.Set[str], traffic:Traffic, expected_speed:float=200, threshold_shift:float=20):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    traffic = traffic.compute_xy()
    
    
    # Setup colors for plotting
    cmap = cm["tab10"]
    cmap_i = 0
    cmap_div = 10
    colordict:dict[tuple[str,str],Colortype] = dict()
    
    plotnames = ('Start','End','Estimated IF')
    markers = ('o','x','s')
    linestyles = ('-','-',':')
    shifts = (expected_speed/60,expected_speed/60,threshold_shift)
    
    for i,flight in enumerate(traffic):
        flight.map_leaflet()
        stats = ACStats(
            i,
            expected_speed/60, # Convert from kts (NM/h) to NM/minute
            1.,
            expected_speed/(60*np.pi) # Full circle in 2 minutes
        )
        
        r = extract_flight_endpoints(flight,ICAO_set,stats,threshold_shift)
        if r is None:
            continue
        else:
            if (r[0].dest_ICAO,r[0].dest_runway) not in colordict.keys():
                colordict[(r[0].dest_ICAO,r[0].dest_runway)] = cmap(cmap_i/cmap_div)
                cmap_i = (cmap_i+1) % cmap_div
            
            color = colordict[(r[0].dest_ICAO,r[0].dest_runway)]
                
            start       = r[0].start
            end         = r[0].end
            shifted_thr = r[1].end
            
            for p,m,ls,dp in zip([start,end,shifted_thr],markers,linestyles,shifts):
                ax.scatter(
                    [p.longitude],[p.latitude],s=10,color=color,marker=m
                )
                
                e = destination(p.latitude,p.longitude,p.bearing,dp * NM_TO_METERS)
                
                ax.plot(
                    [p.longitude,e[1]],
                    [p.latitude,e[0]],
                    linestyle=ls,
                    color=color
                )
    
    
    
    for (a,rw),c in colordict.items():
        ax.plot([],[],marker='s',linestyle='',color=c,label=f"{a} ({rw})")
        
    for m,ls,n in zip(markers,linestyles,plotnames):
        ax.plot([],[],marker=m,linestyle=ls,label=n,color=(0.2,0.2,0.2,0.5))
    
    ax.legend()
    fig.tight_layout()
    plt.show()
                

#################### Entrypoint ####################

def main():
    ICAO_set = set(["LFPO","LFPG","LFPB"])
    try:
        traffic = Traffic.from_file('benchdata.parquet')
        filtered = True
    except FileNotFoundError:
        traffic = samples.quickstart
        filtered = False
    if traffic is None:
        print("ERROR: Importing traffic failed. Exiting")
        exit(1)
    
    # Filtering to remove aicraft not landing
    if not(filtered):
        print("Filtering dataset...")
        flight_to_remove = set()
        for flight in traffic:
            if flight_landing(flight,ICAO_set) is None:
                flight_to_remove.add(flight.icao24)

        df = traffic.data
        traffic = Traffic(df[~df.icao24.isin(flight_to_remove)]).compute_xy().filter(filters.FilterAboveSigmaMedian()).eval(8) # type: ignore
        if traffic is None:
            print("ERROR: Filtering turned the database into None")
            exit(1)
        
            
        traffic.to_parquet('benchdata.parquet')
        
    plot(ICAO_set, traffic)



if __name__ == '__main__':
    main()
