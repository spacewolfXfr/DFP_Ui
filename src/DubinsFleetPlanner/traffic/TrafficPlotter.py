#!/usr/bin/python3

from __future__ import annotations

import pathlib
import typing
import pickle

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.axes import Axes
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

import pandas as pd

from pitot.geodesy import destination

from traffic.core import Traffic
from traffic.data import samples

from cartopy import crs as ccrs

from .FlightExtraction import generate_flightEndpoints,generate_both_flightEndpoints,NM_TO_METERS,FlightEndpoints
from .TrafficReader import filter_traffic,iter_flight_by_period
from .AirportHelpers import get_airports,Airports
from .TaskResultsPlotter import SimulationSummary

from DubinsFleetPlanner.UI.plotting import ColorType



##################### Helpers ####################

colordict_counter = 0

def make_colordict(tasks:list[FlightEndpoints], colormap:str|typing.Callable[[float], ColorType]='tab10',color_period:int=10, existing:dict[tuple[str,str,bool],ColorType]=dict()) -> typing.Tuple[dict[tuple[str,str,bool],ColorType],dict[str,ColorType]]:
    global colordict_counter
    runway_dict:dict[tuple[str,str,bool],ColorType] = existing
    id_dict:dict[str,ColorType] = dict()
    if isinstance(colormap,str):
        my_cm = cm[colormap]
    else:
        my_cm = colormap
    i = colordict_counter
    for t in tasks:
        if not t.departing:
            assert t.dest_ICAO is not None
            try:
                color = runway_dict[(t.dest_ICAO,t.dest_runway,False)]
            except KeyError:
                color = my_cm(i/color_period)
                i+=1
                runway_dict[(t.dest_ICAO,t.dest_runway,False)] = color
        else:
            assert t.start_ICAO is not None
            try:
                color = runway_dict[(t.start_ICAO,t.start_runway,True)]
            except KeyError:
                color = my_cm(i/color_period)
                i+=1
                runway_dict[(t.start_ICAO,t.start_runway,True)] = color
        
        if t.icao24 in id_dict.keys():
            assert id_dict[t.icao24] == color
        id_dict[t.icao24] = color
    colordict_counter = i
    return runway_dict, id_dict


#################### Plotting ####################

def animate(airports:Airports, traffic:Traffic, speedup:float=60,
            trace_size:int=50, expected_speed:float=200,
            threshold_shift:float=20, show:bool=True,
            colordict:dict[tuple[str,str,bool],ColorType]=dict()) -> tuple[FuncAnimation,dict[tuple[str,str,bool],ColorType]]:
    """Animate traffic data with ``speedup`` times the source time scale.

    ``speedup=60`` means that one minute of source data takes one second to
    replay when the source sample interval is one second. The returned
    animation must be kept alive by the caller when ``show`` is false.
    """
    if speedup <= 0:
        raise ValueError("speedup must be greater than zero")
    if trace_size < 1:
        raise ValueError("trace_size must be greater than zero")

    data = traffic.data.copy()
    if data.empty:
        raise ValueError("traffic contains no data")
    required_columns = {"timestamp", "latitude", "longitude"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"traffic is missing columns: {sorted(missing_columns)}")

    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values("timestamp")
    timestamps = data["timestamp"].drop_duplicates().to_numpy()
    time_deltas = data["timestamp"].diff().dropna().dt.total_seconds()
    sample_seconds = float(time_deltas[time_deltas > 0].median()) if len(time_deltas[time_deltas > 0]) else 1.
    interval_ms = max(1, sample_seconds * 1000 / speedup)

    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    cmap = cm["tab10"]
    flights = list(traffic)
    default_color = (0.2,0.2,0.2,0.5)
    plotnames = ('Start','End','Sim start', 'Sim end')
    markers = ('o','x','s','d')
    linestyles = ('-','-',':',':')
    shifts = (expected_speed/60,expected_speed/60,expected_speed/60,expected_speed/60)
    
    endpoints = generate_both_flightEndpoints(airports,expected_speed,expected_speed,traffic,True,None)
    
    colordict, id_colordict = make_colordict([e[1] for e in endpoints.values() if e[1] is not None],existing=colordict)

    for _, flight in enumerate(flights):
        color = id_colordict.get(str(flight.flight_id), default_color)
        flight.plot(ax, color=color, alpha=0.25)
        endpts = endpoints.get(str(flight.flight_id),None)
        if endpts is not None:
            points = [endpts[0].start, endpts[0].end]
            if endpts[1] is not None:
                points.append(endpts[1].start)
                points.append(endpts[1].end)
            for point, marker, shift in zip(points, markers,shifts):
                ax.scatter([point.longitude], [point.latitude], s=10, color=color, marker=marker)
                projected = destination(point.latitude, point.longitude, point.bearing, shift * NM_TO_METERS)
                ax.plot([point.longitude, projected[1]], [point.latitude, projected[0]], color=color, linestyle=":" if marker == "s" else "-")

    id_column = "flight_id" if "flight_id" in data.columns else "icao24"
    data["_animation_id"] = data[id_column].astype(str)
    flight_ids = data["_animation_id"].unique()
    position_lines:dict[str,Line2D] = {}
    trace_lines:dict[str,Line2D] = {}
    for i, flight_id in enumerate(flight_ids):
        color = colors.get(flight_id, cmap(i % 10 / 10))
        trace_lines[flight_id] = ax.plot([], [], linestyle="-", alpha=0.5, color=color)[0]
        position_lines[flight_id] = ax.plot([], [], marker="o", linestyle="", color=color, label=flight_id)[0]

    ax.legend()
    fig.tight_layout()
    start_time = pd.Timestamp(timestamps[0])

    def update(frame_index:int) -> list[Line2D]:
        current_time = pd.Timestamp(timestamps[frame_index])
        frame_data = data[data["timestamp"] <= current_time]
        visible_artists:list[Line2D] = []
        for flight_id in flight_ids:
            flight_data = frame_data[frame_data["_animation_id"] == flight_id].tail(trace_size)
            trace = trace_lines[flight_id]
            position = position_lines[flight_id]
            if flight_data.empty:
                trace.set_visible(False)
                position.set_visible(False)
                continue
            trace.set_data(flight_data["longitude"], flight_data["latitude"])
            latest = flight_data.iloc[-1]
            position.set_data([latest["longitude"]], [latest["latitude"]])
            trace.set_visible(True)
            position.set_visible(True)
            visible_artists.extend((trace, position))
        elapsed = current_time - start_time
        ax.set_title(f"Traffic replay: {current_time} (+{elapsed})")
        return visible_artists

    animation = FuncAnimation(fig, update, frames=len(timestamps), interval=interval_ms, blit=False, repeat=False)
    if show:
        plt.show()
    return animation,colordict

def plot(airports:Airports, traffic:Traffic, expected_speed:float=200, endpoints:dict[str,FlightEndpoints]|None=None, colordict:dict[tuple[str,str,bool],ColorType]=dict()) -> dict[tuple[str,str,bool],ColorType]:
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Setup colors for plotting
    if colordict is None:
        colordict = dict()
    default_color = (0.2,0.2,0.2,0.5)
    
    plotnames = ('Start','End','Sim start', 'Sim end')
    markers = ('o','x','s','d')
    linestyles = ('-','-',':',':')
    shifts = (expected_speed/60,expected_speed/60,expected_speed/60,expected_speed/60)
    
    if endpoints is None:
        endpoints = generate_flightEndpoints(airports,expected_speed,expected_speed,traffic,True,None,True)    
    
    colordict, id_colordict = make_colordict(list(endpoints.values()),existing=colordict)
        
    for flight in traffic:
        if flight.icao24 is None:
            continue
        assert isinstance(flight.icao24,str)
        
    
        try:
            r = endpoints[flight.icao24]
            color = id_colordict[flight.icao24]
        except KeyError:    
            flight.plot(ax,c=default_color,alpha=0.3)
            continue
        
            
        flight.plot(ax,color=color,alpha=0.3)
        start       = r.start
        end         = r.end
        pts = [start,end]
        
        for p,m,ls,dp,n in zip(pts,markers,linestyles,shifts,plotnames):
            ax.scatter(
                [p.longitude],[p.latitude],s=10,color=color,marker=m
            )
            
            try:
                e = destination(p.latitude,p.longitude,p.bearing,dp * NM_TO_METERS)
            except Exception as e:
                print(f"Error at flight {flight} for point {n} with coordinates {p}: {e}\nSkipping this point.")
                continue
            
            ax.plot(
                [p.longitude,e[1]],
                [p.latitude,e[0]],
                linestyle=ls,
                color=color
            )

    
    
    for (a,rw,b),c in colordict.items():
        ax.plot([],[],marker='s',linestyle='',color=c,label=f"{a} ({rw} {'departing' if b else 'arriving'})")
        
    for m,ls,n in zip(markers,linestyles,plotnames):
        ax.plot([],[],marker=m,linestyle=ls,label=n,color=(0.2,0.2,0.2,0.5))
    
    ax.set_title("Traffic plot for {} flights".format(len(traffic)))
    ax.legend()
    fig.tight_layout()
    plt.show()
    
    return colordict
                

#################### Entrypoint ####################

def main():
    import argparse
    
    parser = argparse.ArgumentParser('Traffic plotter','Traffic seen in a traffic dataset.')
    parser.add_argument('--demo',action='store_true',help="Run a demo with a sample dataset.")
    parser.add_argument('dataset',help="Path to a traffic dataset in parquet format.")
    parser.add_argument('ICAOs',nargs='+',
                            help="List of ICAO codes for airports in which to look for landings in the given dataset")
    parser.add_argument('-e','--endpoints',help='Pickle path of the companion summary files (for simulated data), containg endpoints information.',default=None)
    parser.add_argument('-f','--freq-str',dest='freq_str',default='2h',
                        help="Frequency string for grouping the data (e.g., 'D' for daily, 'W' for weekly). Full details at https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases")
    parser.add_argument('-a','--animate',action='store_true',help="Animate the traffic data instead of plotting it.")
    
    args = parser.parse_args()
    
    
    print("ICAO codes to look for landings:",args.ICAOs)
    airports = get_airports(args.ICAOs)
    
    if args.demo:
        print("Running demo with sample dataset...")
        traffic = samples.quickstart
        filtered_traffic = filter_traffic(traffic,airports)
        if filtered_traffic is None:
            print("ERROR: Could not filter the demo traffic")
            exit(1)
        if args.animate:
            animate(airports, filtered_traffic)
        else:
            plot(airports, filtered_traffic)
        exit(0)
    
    dataset_path = pathlib.Path(args.dataset)
    if not(dataset_path.is_file()):
        print("ERROR: Dataset path is not a file")
        exit(1)
    traffic_gen = iter_flight_by_period(args.dataset, freq_str=args.freq_str)
    
    if args.endpoints is not None:
        with open(args.endpoints, 'rb') as f:
            summary = pickle.load(f)
            assert isinstance(summary, SimulationSummary), f"The summary file {args.endpoints} is not a SimulationSummary"
            endpoints = {t.icao24:t for t in summary.tasks}
    else:
        endpoints = None
    
    colordict = dict()
    for traffic in traffic_gen:
        print(f"Processing traffic from {traffic.data['timestamp'].min()} to {traffic.data['timestamp'].max()}")
        filtered_traffic = filter_traffic(traffic,airports)
        if filtered_traffic is None:
            print("ERROR: Could not filter the day of traffic")
            continue
        if args.animate:
            _, colordict = animate(airports, filtered_traffic, colordict=colordict)
        else:
            colordict = plot(airports, filtered_traffic, endpoints=endpoints, colordict=colordict)
        cont = input("Enter 'y' to start simulation with the next period, quit otherwise: ")
        if cont != 'y':
            break


if __name__ == '__main__':
    main()
