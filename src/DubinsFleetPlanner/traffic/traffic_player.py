#!/usr/bin/python3

import time

import dataclasses
import typing

import csv,pathlib

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib import colormaps as cm

from traffic.core import Traffic,Flight
from traffic.data import samples,airports
from traffic.algorithms import filters

from pitot.geodesy import distance
from pyproj import Geod

from .traffic.traffic_pb_generator import NM_TO_METERS

geod = Geod(ellps="WGS84")

ColorType = tuple[float,float,float,float]

@dataclasses.dataclass
class FlightArtists:
    path:Line2D
    position:Line2D
    proj_line:Line2D
    alt_line:typing.Optional[Line2D]

#################### ICAO database shenanigans ####################

ICAO_set = set()
airport_db_loc = pathlib.Path(__file__).parent / "airports.csv"
with open(airport_db_loc,'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    assert header[1].lower() == 'icao'
    for l in reader:
        ICAO_set.add(l[1])

def get_arrival_airport(flight:Flight):
    """
    Try multiple ways to infer arrival airport.
    Returns ICAO code or 'UNKNOWN'.
    """
    global ICAO_set
    
    if hasattr(flight, "destination") and flight.destination:
        return flight.destination

    if hasattr(flight, "arrival") and flight.arrival is not None:
        if hasattr(flight.arrival, "airport"):
            return flight.arrival.airport.icao

    
    for icao in ICAO_set:
        try:
            if flight.landing_at(icao):
                return icao
        except ValueError:
            continue
    
    return "UNKNOWN"

def known_arrival_airport(flight:Flight) -> bool:
    return get_arrival_airport(flight) != "UNKNOWN"

#################### Other helpers ####################

def project_position(lon:float, lat:float, track_deg:typing.Optional[float], groundspeed_kt:typing.Optional[float], dt_sec:float=60):
    """
    Project aircraft position forward using track and groundspeed.

    Returns (lon2, lat2)
    """
    if groundspeed_kt is None or track_deg is None:
        return None
    
    try:
        if not(np.isfinite(groundspeed_kt) and np.isfinite(track_deg)):
            return None
    except TypeError:
        return None

    # Convert knots → meters per second
    speed_mps = groundspeed_kt * 0.514444
    distance_m = speed_mps * dt_sec

    lon2, lat2, _ = geod.fwd(lon, lat, track_deg, distance_m)
    return lon2, lat2

def compute_min_dist(last_known_pos:dict[typing.Any,tuple[float,float,pd.Timestamp]],ref_t:pd.Timestamp,max_dt:pd.Timedelta=pd.Timedelta("60s")):
    
    positions = list(filter(lambda t : (ref_t - t[2]) <= max_dt,last_known_pos.values()))
    
    if len(positions) <= 1:
        return None
    min_p1 = None
    min_p2 = None
    min_dist = np.inf
    for i in range(len(positions)):
        for j in range(i+1,len(positions)):
            p1 = positions[i]
            p2 = positions[j]
            dist = distance(p1[0],p1[1],p2[0],p2[1]) / NM_TO_METERS
            
            if dist < min_dist:
                min_p1 = p1
                min_p2 = p2
                min_dist = dist
                
    return min_p1,min_p2,min_dist

#################### Replayer class ####################

@dataclasses.dataclass
class TrafficReplay:
    traffic:Traffic
    trace_size:int = 50
    trace_interval:int = 20
    fig:typing.Optional[Figure]     = None
    ax:typing.Optional[Axes]        = None
    ax_alt:typing.Optional[Axes]    = None
    artists:dict[str,FlightArtists] = dataclasses.field(init=False)
    filling_index:dict[str,int]     = dataclasses.field(init=False)
    rolling_index:dict[str,int]     = dataclasses.field(init=False)
    arrival_by_icao24:dict[str,str] = dataclasses.field(init=False)
    airports_colors:dict[str,ColorType] = dataclasses.field(init=False)
    
    max_alt:float = 1
    
    __mdist_line:typing.Optional[Line2D] = dataclasses.field(init=False)
    __legend_handles:list           = dataclasses.field(init=False)
    
    def __post_init__(self):
        self.filling_index      = dict()
        self.rolling_index      = dict()
        self.arrival_by_icao24  = dict()
        self.airports_colors    = dict()
        self.artists            = dict()
        self.__legend_handles   = list()
        
        cmap = cm.get_cmap("tab10")
        cmap_index = 0
        
        if self.ax is not None:
            self.__mdist_line = self.ax.plot([],[],linestyle=':',color='k',marker='D',label="Min dist:")[0]
            
            self.__legend_handles = [self.__mdist_line]
            
            for flight in self.traffic:
                airport = get_arrival_airport(flight)
                print(flight,airport)
                self.arrival_by_icao24[flight.icao24] = airport
                
                color = self.airports_colors.get(airport)
                if color is None:
                    color = cmap(cmap_index/9)
                    cmap_index += 1
                    self.airports_colors[airport] = color
                
                
                (line,) = self.ax.plot(np.zeros(self.trace_size),np.zeros(self.trace_size),":",alpha=0.5,color=color)
                
                (dot,) = self.ax.plot([],[], "o", markersize=3, label=flight.callsign,color=color)
                
                (proj_line,) = self.ax.plot(
                    [], [],
                    linestyle="-",
                    linewidth=1,
                    color=color,
                    alpha=0.7,
                    zorder=3)
                
                self.filling_index[flight.icao24] = 0
                self.rolling_index[flight.icao24] = 0
                
                if self.ax_alt is not None:
                    (alt_line,) = self.ax_alt.plot(
                        [],
                        [],
                        color=color,
                        linewidth=1.5,
                        alpha=0.8
                    )
                else:
                    alt_line = None

                
                artist_container = FlightArtists(
                    line,dot,proj_line,alt_line
                )
                self.artists[flight.icao24] = artist_container
            
            
            for k,v in self.airports_colors.items():
                if k == "UNKNOWN":
                    continue
                
                ap = airports[k]
                lon = ap.longitude
                lat = ap.latitude
                
                self.__legend_handles.append(Line2D([0], [0], color=v, marker="o", linestyle="-",label=f"{ap.icao}"))
                
                self.ax.plot(
                    lon,
                    lat,
                    marker="^",
                    markersize=10,
                    color=v,
                    markeredgecolor="black",
                    zorder=5
                )

                self.ax.text(
                    lon,
                    lat,
                    f"  {k}",
                    fontsize=9,
                    weight="bold",
                    verticalalignment="center",
                    zorder=6
                )

            self.ax.legend(handles=self.__legend_handles)
                
            self.ax.set_xlim(self.traffic.data.longitude.min() - 1, self.traffic.data.longitude.max() + 1)
            self.ax.set_ylim(self.traffic.data.latitude.min() - 1, self.traffic.data.latitude.max() + 1)

    def replay(self,
        speed:float,
        on_update=None,
    ):
        """
        Replay aircraft trajectories in real time.

        Parameters
        ----------
        on_update : callable
            Function called for each aircraft state update
        """

        # Flatten all points into one timeline
        df = (
            self.traffic
            .data
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        start_data_time = df.timestamp.iloc[0]
        end_data_time = df.timestamp.iloc[-1]
        
        if self.ax_alt is not None:
            self.ax_alt.set_xlim(start_data_time,end_data_time)
        
        start_wall_time = time.time()
        last_know_pos = dict()

        for _, row in df.iterrows():
            # Compute how much simulated time has passed
            sim_elapsed = (row.timestamp - start_data_time).total_seconds()
            wall_elapsed = (time.time() - start_wall_time) * speed
            
            last_know_pos[row.flight_id] = (row.latitude,row.longitude,row.timestamp)

            # Wait if simulation is ahead of wall clock
            if sim_elapsed > wall_elapsed:
                time.sleep((sim_elapsed - wall_elapsed) / speed)

            # Emit update
            if on_update:
                on_update(row)

            # Update plot
            lon, lat = row.longitude, row.latitude
            
            flight_artists = self.artists.get(row.icao24)
            
            if self.ax is None or flight_artists is None:
                continue
                        
            ## Title
            self.ax.set_title(f"Sim time: {row.timestamp}")
            
            ## Position
            flight_artists.position.set_data([lon], [lat])
            
            ## Projection
            proj = flight_artists.proj_line
            try:
                on_ground = bool(row.altitude < 1000 or row.groundspeed < 50)
            except TypeError:
                on_ground = False
                
            if on_ground:
                proj.set_data([], [])
            else:
                projected = project_position(
                    lon,
                    lat,
                    row.track,
                    row.groundspeed,
                    dt_sec=60
                )
                if projected is not None:
                    
                    lon2, lat2 = projected
                    proj.set_data([lon, lon2], [lat, lat2])
                else:
                    proj.set_data([], [])
            
            ## Trace
            line = flight_artists.path
            stepping_index = self.rolling_index.get(row.icao24)
            last_registered_index = self.filling_index.get(row.icao24)
            if line is None or stepping_index is None or last_registered_index is None:
                continue
            
            
            if stepping_index == 0:
                x, y = line.get_data()
                if last_registered_index < self.trace_size:
                    x[last_registered_index:] = lon
                    y[last_registered_index:] = lat
                    self.filling_index[row.icao24] = last_registered_index+1
                else:
                    x[0] = lon
                    y[0] = lat
                    x = np.roll(x,-1)
                    y = np.roll(y,-1)
                
                line.set_data(x,y)
            else:
                self.rolling_index[row.icao24] = (stepping_index+1) % self.trace_interval
            
            ## Altitude
            if self.ax_alt is not None and flight_artists.alt_line is not None:
                alt_line = flight_artists.alt_line
                ts,alts = alt_line.get_data()
                
                try:
                    # self.max_alt = max(self.max_alt,row.altitude)
                    self.max_alt = max(self.max_alt,row.groundspeed)
                    # alt_line.set_data(np.append(ts,[row.timestamp]),np.append(alts,[row.altitude]))
                    alt_line.set_data(np.append(ts,[row.timestamp]),np.append(alts,[row.groundspeed]))
                    
                    left_t,_ = self.ax_alt.get_xlim()
                    if (left_t-row.timestamp).seconds > 10:
                        self.ax_alt.set_xlim(right=row.timestamp)
                    self.ax_alt.set_ylim(0, self.max_alt)
                except TypeError:
                    pass
            
            ## Min dist
            if self.ax is not None and self.__mdist_line is not None:
                mdist_data = compute_min_dist(last_know_pos,row.timestamp)
                if mdist_data is not None:
                    p1,p2,dist = mdist_data
                    self.__mdist_line.set_xdata(
                        [p1[0],p2[0]]
                    )
                    self.__mdist_line.set_ydata(
                        [p1[1],p2[1]]
                    )
                    self.__mdist_line.set_label(f"Min dist: {dist:.1f} NM")
                    print(mdist_data[2])
            
            if self.ax is not None:
                self.ax.legend(handles=self.__legend_handles)
                
            plt.pause(0.001)


#################### Entry point ####################


def main(data_file:str, speed:float, filter:bool=True):
    global ICAO_set
    
    ICAO_set = set(["LFPO","LFPG","LFPB"])
    # traffic = Traffic.from_file("")
    
    print("Loading dataset...")
    traffic = Traffic.from_file(data_file)
    
    if filter:
        print("Filtering dataset...")
        flight_to_remove = set()
        for flight in traffic:
            if get_arrival_airport(flight) == "UNKNOWN":
                flight_to_remove.add(flight.icao24)
        df = traffic.data
        traffic = Traffic(df[~df.icao24.isin(flight_to_remove)]).compute_xy().filter(filters.FilterAboveSigmaMedian()).eval(8)
        if traffic is None:
            print("ERROR: Filtering turned the database into None")
            exit(1)
            
        traffic = traffic.assign_id().eval(8)
        # traffic.to_parquet('testdata.parquet')
    
    if traffic is None:
        print("isded")
        exit(1)
    ##### Matplotlib setup #####
    
    plt.ion()  # interactive mode

    fig, (ax,ax_alt) = plt.subplots(2,1,
                        figsize=(12, 10),
                        gridspec_kw={"height_ratios": [3, 1]},
                        sharex=False)
    ax.set_title("Real-time Aircraft Trajectory Replay")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    
    # ax_alt.set_title("Altitude profile (by destination)")
    ax_alt.set_title("Ground speed profile (by destination)")
    ax_alt.set_xlabel("Time")
    # ax_alt.set_ylabel("Altitude (ft)")
    ax_alt.set_ylabel("Ground speed (kt)")
    ax_alt.grid(True)

    # Reasonable defaults (auto-adjust later)
    ax_alt.set_ylim(0, traffic.data.altitude.max() * 1.1)
    
    traffic = traffic.resample(100).eval(8)
    
    trafficRP = TrafficReplay(traffic,100,100,fig=fig,ax=ax,ax_alt=ax_alt)
    trafficRP.replay(speed)
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Replay aircraft trajectories in real time.")
    parser.add_argument("file", type=str, help="Path to the Parquet file containing the traffic data.")
    parser.add_argument("--speed", type=float, default=60.0, help="Replay speed factor (default: 60.0, meaning 1 minute of simulation time per second of wall clock time).")
    parser.add_argument('--ICAOs',nargs='+',
                    help="List of ICAO codes for airports in which to look for landings in the given dataset. Default to LFPO,LFPG,LFPB",
                    default=["LFPO","LFPG","LFPB"])
    
    args = parser.parse_args()
    
    icao_set = set(args.ICAOs)
    file = args.file 
    speed = args.speed
    
    main(file,speed,True)
    
    
    
