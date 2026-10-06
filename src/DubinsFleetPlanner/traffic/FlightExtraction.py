from __future__ import annotations

import typing
from typing import Optional
import dataclasses
import numpy as np

import pandas as pd

from traffic.core import Flight
from traffic.core.structure import Airport
from traffic.data.basic.airports import Airports

from pyproj import Transformer

from pitot.geodesy import bearing, destination, distance


from DubinsFleetPlanner.Poses import Pose3D
from DubinsFleetPlanner.ioUtils import AC_PP_Problem, ACStats

from .AirportHelpers import get_airport, get_airports, get_runway, InfluenceCircle, NM_TO_METERS

#################### Pose as Latlon ####################

@dataclasses.dataclass
class LatlonPose:
    latitude:float
    longitude:float
    altitude:float  # in fts
    bearing:float   # in degrees
    
    @staticmethod
    def from_pose3D(pose:Pose3D,transformer:Transformer, from_NM:bool=False) -> LatlonPose:
        if from_NM:
            lat,lon = transformer.transform(pose.x*NM_TO_METERS,pose.y*NM_TO_METERS,direction='INVERSE')
        else:
            lat,lon = transformer.transform(pose.x,pose.y,direction='INVERSE')
        return LatlonPose(lat,lon,pose.z,np.rad2deg(90-pose.theta))
    
    def to_pose3D(self,transformer:Transformer,to_NM:bool=False) -> Pose3D:
        x,y = transformer.transform(self.latitude,self.longitude)
        if to_NM:
            return Pose3D(x/NM_TO_METERS,y/NM_TO_METERS,self.altitude,np.deg2rad(90-self.bearing))
        else:
            return Pose3D(x,y,self.altitude,np.deg2rad(90-self.bearing))
        
    def project(self,distance:typing.Annotated[float,"m"]) -> LatlonPose:
        nlat,nlon,ndir = destination(self.latitude,self.longitude,self.bearing,distance)
        return LatlonPose(nlat,nlon,self.altitude,ndir+180)
    
    def distance(self,other:LatlonPose) -> float:
        return distance(self.latitude,self.longitude,other.latitude,other.longitude)

def flight_datapoint_to_pose(dpt:pd.Series) -> LatlonPose:
    return LatlonPose(
        dpt["latitude"],
        dpt["longitude"],
        dpt["altitude"],
        dpt["track"]
    )

#################### Endpoint detection ####################

def flight_landing(flight:Flight,candidate_airports:typing.Iterable[str]|Airports,maxdist:float=1000.) -> Optional[Airport]:
    """ Returns where the aircraft lands among the candidates, if the guessed airport is closed than the given threshold

    Args:
        flight (Flight): Input flight
        candidate_airports (typing.Iterable[str]): List of candidate airports (ICAO codes) where the aircraft could have landed
        maxdist (float, optional): Maximum allowed distance from the guessed airport, in meters. Defaults to 1000m

    Returns:
        Optional[Airport]: An airport if the best guess is within the threshold, None otherwise
    """
    
    if not isinstance(candidate_airports,Airports):
        candidate_airports = get_airports(candidate_airports)
    output = flight.infer_airport("landing", dataset=candidate_airports)
    if output.distance > maxdist:
        return None
    else:
        return output

def flight_departing(flight:Flight,candidate_airports:typing.Iterable[str]|Airports,maxdist:float=1000.) -> Optional[Airport]:
    """ Returns where the aircraft takes off among the candidates, if the guessed airport is closed than the given threshold
    
        Args:
            flight (Flight): Input flight
            candidate_airports (typing.Iterable[str]): List of candidate airports (ICAO codes) where the aircraft could have taken off
            maxdist (float, optional): Maximum allowed distance from the guessed airport, in meters. Defaults to 1000m
    
        Returns:
            Optional[Airport]: An airport if the best guess is within the threshold, None otherwise
        """
    if not isinstance(candidate_airports,Airports):
        candidate_airports = get_airports(candidate_airports)
    output = flight.infer_airport("takeoff", dataset=candidate_airports)
    if output.distance > maxdist:
        return None
    else:
        return output

def get_ils_aligned_datapoint(flight:Flight,airport:str|Airport) -> Optional[pd.Series]:
    ends = flight.landing(airport,method='aligned_on_ils').next()
    if ends is None:
        return None
    fst_timestamp = ends.data["timestamp"].min() # type: ignore
    return ends.data[ends.data["timestamp"] == fst_timestamp].iloc[0]

def get_ils_second_aligned_datapoint(flight:Flight,airport:str|Airport) -> Optional[pd.Series]:
    ends = flight.landing(airport,method='aligned_on_ils').next()
    if ends is None:
        return None
    ends_sorted = ends.data.sort_values("timestamp")
    if len(ends_sorted) < 2:
        return None
    else:
        return ends_sorted.iloc[1]
    
def get_any_landing_datapoint(flight:Flight) -> Optional[pd.Series]:
    ends = flight.landing(method='anywhere').next()
    if ends is None:
        return None
    fst_timestamp = ends.data["timestamp"].min() # type: ignore
    return ends.data[ends.data["timestamp"] == fst_timestamp].iloc[0]

def get_ils_last_aligned_datapoint(flight:Flight,airport:str|Airport) -> Optional[pd.Series]:
    ends = flight.landing(airport,method='aligned_on_ils').next()
    if ends is None:
        return None
    lst_timestamp = ends.data["timestamp"].max() # type: ignore
    return ends.data[ends.data["timestamp"] == lst_timestamp].iloc[0]

def get_first_airbone_dpt(flight:Flight) -> Optional[pd.Series]:
    airborne = flight.airborne()
    if airborne is None:
        return None
    fst_timestamp = airborne.data["timestamp"].min() # type: ignore
    return airborne.data[airborne.data["timestamp"] == fst_timestamp].iloc[0]

def get_second_airbone_dpt(flight:Flight) -> Optional[pd.Series]:
    airborne = flight.airborne()
    if airborne is None:
        return None
    sorted = airborne.data.sort_values("timestamp")
    if len(sorted) < 2:
        return None
    else:
        return sorted.iloc[1]


@dataclasses.dataclass
class FlightEndpoints:
    start:LatlonPose
    start_time:pd.Timestamp
    end:LatlonPose
    end_time:pd.Timestamp
    initial_end_time:pd.Timestamp
    stats:ACStats   # Speed is expected in NM / minute, turn radius in NM
    start_airport:Optional[Airport] = None
    start_runway:Optional[str]      = None
    dest_airport:Optional[Airport]  = None
    dest_runway:Optional[str]       = None
    planned:bool                    = False
    
    @property
    def id(self) -> int:
        return self.stats.id
    
    @id.setter
    def id(self,_id:int):
        self.stats.id = _id
        
    @property
    def start_ICAO(self) -> Optional[str]:
        return self.start_airport.icao if self.start_airport is not None else None    
    
    @property
    def dest_ICAO(self) -> Optional[str]:
        return self.dest_airport.icao if self.dest_airport is not None else None

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
            self.stats,
            self.start_airport,
            self.start_runway,
            self.dest_airport,
            self.dest_runway,
            self.planned
        )
    
    def to_AC_PP_Problem(self,transformer:Transformer,timeshifts:list[pd.Timedelta],flatten:bool=True, add_initial_time:bool=False) -> AC_PP_Problem:
        """
        Export from this flight endpoints an Aircraft Path Planning problem  
        
        
        Args:
            transformer (Transformer): Convert latlon to a flat projection. Assumes final projection is in meters (which are then converted in NM)
            timeshifts (list[pd.Timedelta]): Allowed time differences with respect to current time of arrival (different planning objectives)
            flatten (bool, optional): Set all vertical values to 0. Defaults to True.

        Returns:
            AC_PP_Problem: _description_
        """
        self.start.to_pose3D(transformer,True)
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
        if add_initial_time:
            for ts in timeshifts:
                slot = ts + (self.initial_end_time - self.start_time)
                if slot.total_seconds() >= 0:
                    timeslots.append(slot.total_seconds()/60)
        
        return AC_PP_Problem(
            self.stats,
            start_proj,
            end_proj,
            timeslots=timeslots
        )

def extract_flight_endpoints(flight:Flight,candidate_airports:typing.Iterable[str]|Airports,
                             stats:ACStats,circle:Optional[InfluenceCircle]=None,
                             threshold_shift:float=20) -> Optional[tuple[FlightEndpoints,Optional[FlightEndpoints]]]:
    """ Given a flight, a list of airports where it could have started or landed (ICAO codes) and a threshold shift, returns:
    - None if neither takeoff nor landing airport could be confirmed among the candidates ones
    - A pair a FlightEndpoints. The first one is from the start to the end of the recorded trajectory.
        The second depends on if it is a departing or arriving flight. In the departing case, the endpoint is set to the influence circle limit (if provided).
        In the arriving case, it is from start to the landing runway threshold, shifted by `threshold_shift`. If the landing runway could not be confirmed, the second FlightEndpoints is None.

    Args:
        flight (Flight): Flight object
        stats (ACStats): Assumed characteristics of the aircraft
        candidate_airports (typing.Iterable[str]|Airports): List of airports where the aircraft possibly landed
        circle (InfluenceCircle): The influence circle around the airports
        threshold_shift (float, optional): Distance to pre-shift from the landing runway threshold, in Nautical Miles. Defaults to 20 NM.

    Returns:
        Optional[tuple[FlightEndpoints,FlightEndpoints]]: None if no landing could be confirmed, or a pair of endpoints. The first is from start to end of the recording, the second is adapted to the type of trajectory (departing or arriving).
    """
    if not isinstance(candidate_airports,Airports):
        candidate_airports = get_airports(candidate_airports)
    try:
        start_airport = flight.data['departing_airport'].iloc[0]
        if start_airport is not None:
            start_airport = get_airport(start_airport)
    except Exception as e:
        print(f"Error extracting departing airport for flight {flight}: {e}")
        start_airport = flight_departing(flight,candidate_airports)
        
    try:
        dest_airport = flight.data['landing_airport'].iloc[0]
        if dest_airport is not None:    
            dest_airport = get_airport(dest_airport)
    except Exception as e:
        print(f"Error extracting landing airport for flight {flight}: {e}")
        dest_airport = flight_landing(flight,candidate_airports)
        
    flight.data['track'] = flight.data['track'].fillna(flight.data.pop('compute_track'))
    flight = Flight(flight.data.dropna(subset=["latitude","longitude","altitude","track"]))
    # print(f"Flight endpoints extraction: {len(flight.data)} points after dropping NaN values.\n\tStart airport: {start_airport}\n\tDestination airport: {dest_airport}")
    
    
    fix_dpt = None
    
    ## Check for landing point in the arriving case
    runway  = None
    rw_proj = None
    
    if dest_airport is not None and dest_airport.icao in candidate_airports.data.icao.values:
        fix_dpt = get_ils_second_aligned_datapoint(flight,dest_airport)
        fix_dpt = fix_dpt if fix_dpt is not None else get_ils_aligned_datapoint(flight,dest_airport)
        try:
            fix_dpt = fix_dpt if fix_dpt is not None else get_any_landing_datapoint(flight)
        except Exception as e:
            print(f"Error extracting any landing datapoint for flight {flight}: {e}")
            fix_dpt = None
        # last_dpt = get_ils_last_aligned_datapoint(flight,dest_airport)
        if fix_dpt is not None:
            runway = get_runway(dest_airport,fix_dpt["ILS"])
            # print(f"Flight {flight} is landing at airport {dest_airport.icao} on runway {runway.name} (ILS: {fix_dpt['ILS']})")
            if runway is not None:
                # Turn around and convert NM to meters
                rw_proj = destination(runway.latitude, runway.longitude, runway.bearing + 180, threshold_shift * NM_TO_METERS)
            else:
                print(f"Flight {flight} is landing at airport {dest_airport.icao} on ILS {fix_dpt['ILS']} but no runway could be found.")
        else:
            print(f"Flight {flight} is landing at airport {dest_airport.icao} but no ILS-aligned point could be found.")
    
    
    ## Check for takeoff point in the departing case, as well as the influence circle limit if provided
    first_dpt = None
    start_runway = None
    if start_airport is not None and start_airport.icao in candidate_airports.data.icao.values:
        first_dpt = get_second_airbone_dpt(flight)
        takeoff = flight.takeoff(start_airport,method="track_based").next()
        if takeoff is not None:
            start_runway = takeoff.runway_max
        
        if circle is not None:
            # Find the first point outside the influence circle
            prev_row = None
            for _,row in flight.data.iterrows():
                if prev_row is not None:
                    d = distance(row["latitude"],row["longitude"],circle.lat,circle.lon)
                    if d > circle.radius*NM_TO_METERS:
                        fix_dpt = prev_row
                        break
                prev_row = row

    flight.data.sort_values("timestamp",inplace=True)
    if first_dpt is None:
        first_dpt = flight.data.iloc[1]
    
    last_dpt = flight.data.iloc[-2]
    
    dest_runway = None
    if fix_dpt is not None:
        try:
            dest_runway = fix_dpt["ILS"]
        except KeyError:
            pass
    
    end_to_end = FlightEndpoints(
        flight_datapoint_to_pose(first_dpt),
        first_dpt["timestamp"],
        flight_datapoint_to_pose(last_dpt),
        last_dpt["timestamp"],
        last_dpt["timestamp"],
        stats,
        start_airport,start_runway,
        dest_airport,dest_runway,
    )
    
    end_to_ils = None
    if fix_dpt is not None:
        if rw_proj is not None:
            end_to_ils = FlightEndpoints(
                flight_datapoint_to_pose(first_dpt),
                first_dpt["timestamp"],
                LatlonPose(rw_proj[0],rw_proj[1],fix_dpt["altitude"],rw_proj[2]),
                fix_dpt["timestamp"],
                fix_dpt["timestamp"],
                stats,
                start_airport,start_runway,
                dest_airport,fix_dpt["ILS"])
        elif fix_dpt is not None:
            end_to_ils = FlightEndpoints(
                flight_datapoint_to_pose(first_dpt),
                first_dpt["timestamp"],
                flight_datapoint_to_pose(fix_dpt),
                fix_dpt["timestamp"],
                fix_dpt["timestamp"],
                stats,
                start_airport,start_runway,
                dest_airport,None)
        
    return (end_to_end,end_to_ils)