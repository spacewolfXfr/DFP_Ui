from __future__ import annotations

import typing
from typing import Optional
import dataclasses
import numpy as np

import pandas as pd

import traffic
from traffic.core import Flight, FlightPlan
from traffic.core.structure import Airport
from traffic.data import airports
from traffic.data.basic.airports import Airports

from pyproj import Transformer

from pitot.geodesy import bearing, destination, distance


from DubinsFleetPlanner.Poses import Pose3D
from DubinsFleetPlanner.ioUtils import AC_PP_Problem, ACStats

from .AirportHelpers import get_airport, get_airports, get_runway, get_other_runway_name

NM_TO_METERS = 1852

#################### Pose as Latlon ####################

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
    start_airport:Optional[Airport]  = None
    dest_airport:Optional[Airport]   = None
    dest_runway:Optional[str]        = None
    
    @property
    def id(self) -> int:
        return self.stats.id
    
    @id.setter
    def id(self,_id:int):
        self.stats.id = _id
        
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
            self.dest_airport,
            self.dest_runway,
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

def extract_flight_endpoints(flight:Flight,candidate_airports:typing.Iterable[str]|Airports,
                             stats:ACStats,
                             threshold_shift:float=20) -> Optional[tuple[FlightEndpoints,Optional[FlightEndpoints]]]:
    """ Given a flight, a list of airports where it could have started or landed (ICAO codes) and a threshold shift, returns:
    - None if neither takeoff nor landing airport could be confirmed
    - A pair a FlightEndpoints, one from start to first ILS confirmed alignment, the second from start to the landing runway threshold, shifted by `threshold_shift`. If the landing runway could not be confirmed, the second FlightEndpoints is None.

    Args:
        flight (Flight): Flight object
        stats (ACStats): Assumed characteristics of the aircraft
        candidate_airports (typing.Iterable[str]|Airports): List of airports where the aircraft possibly landed
        threshold_shift (float, optional): Distance to pre-shift from the landing runway threshold, in Nautical Miles. Defaults to 20 NM.

    Returns:
        Optional[tuple[FlightEndpoints,FlightEndpoints]]: None if no landing could be confirmed, or a pair of endpoints, from start to either first ILS fix or shifted threshold
    """
    if not isinstance(candidate_airports,Airports):
        candidate_airports = get_airports(candidate_airports)
    # print(f"Extracting flight endpoints for flight {flight} (initial number of points: {len(flight.data)})")
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
    
    ils_dpt = None
    runway  = None
    rw_proj = None
    
    if dest_airport is not None:
        ils_dpt = get_ils_second_aligned_datapoint(flight,dest_airport)
        ils_dpt = ils_dpt if ils_dpt is not None else get_ils_aligned_datapoint(flight,dest_airport)
        if ils_dpt is not None:
            runway = get_runway(dest_airport,ils_dpt["ILS"])
            # print(f"Flight {flight} is landing at airport {dest_airport.icao} on runway {runway.name} (ILS: {ils_dpt['ILS']})")
            if runway is not None:
                # Turn around and convert NM to meters
                rw_proj = destination(runway.latitude, runway.longitude, runway.bearing + 180, threshold_shift * NM_TO_METERS)
            else:
                print(f"Flight {flight} is landing at airport {dest_airport.icao} on ILS {ils_dpt['ILS']} but no runway could be found.")
        else:
            print(f"Flight {flight} is landing at airport {dest_airport.icao} but no ILS-aligned point could be found.")
        
    first_dpt = None
    if start_airport is not None:
        first_dpt = get_second_airbone_dpt(flight)
    
    flight.data.sort_values("timestamp",inplace=True)
    if first_dpt is None:
        first_dpt = flight.data.iloc[1]
    
    last_dpt = flight.data.iloc[-2]
    
    end_to_end = FlightEndpoints(
        flight_datapoint_to_pose(first_dpt),
        first_dpt["timestamp"],
        flight_datapoint_to_pose(last_dpt),
        last_dpt["timestamp"],
        last_dpt["timestamp"],
        stats,
        start_airport,
        dest_airport,ils_dpt["ILS"] if ils_dpt is not None else None,
    )
    
    end_to_ils = None
    if ils_dpt is not None and rw_proj is not None:
        end_to_ils = FlightEndpoints(
            flight_datapoint_to_pose(first_dpt),
            first_dpt["timestamp"],
            LatlonPose(rw_proj[0],rw_proj[1],last_dpt["altitude"],rw_proj[2]),
            last_dpt["timestamp"],
            last_dpt["timestamp"],
            stats,
            start_airport,
            dest_airport,ils_dpt["ILS"])
        
    return (end_to_end,end_to_ils)