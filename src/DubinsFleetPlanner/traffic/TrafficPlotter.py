#!/usr/bin/python3

from __future__ import annotations

import pathlib
import typing
import pickle
import itertools
import os

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.projections.geo import GeoAxes
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib.collections import PathCollection
from matplotlib.artist import Artist
from matplotlib.text import Text

import pandas as pd

from pitot.geodesy import destination

from traffic.core import Traffic
from traffic.data import samples
from traffic.data.basic.airports import Airports,Airport

from cartopy import crs as ccrs

from .FlightExtraction import generate_flightEndpoints,regenerate_flight_id,NM_TO_METERS,FlightEndpoints
from .TrafficReader import filter_traffic,iter_flight_by_period
from .AirportHelpers import get_airports,Airports,get_airport_latlon_transformer
from .TaskResultsPlotter import SimulationSummary

from DubinsFleetPlanner.UI.plotting import ColorType

import tqdm


##################### Helpers ####################

colordict_counter = 0

def make_colordict(tasks:list[FlightEndpoints], colormap:str|typing.Callable[[float], ColorType]='tab10',color_period:int=10, existing:dict[tuple[str,str,bool],ColorType]=dict()) -> typing.Tuple[dict[tuple[str,str,bool],ColorType],dict[str,ColorType]]:
    """ Generate color dictionnaries based on the airport, runway, and behavior (arriving or departing)

    Args:
        tasks (list[FlightEndpoints]): List of tasks to generate colors for
        colormap (str | typing.Callable[[float], ColorType], optional): Colormap to use for generating colors. Defaults to 'tab10'.
        color_period (int, optional): Number of colors in the colormap (number of distinct intervals in the [0,1] range). Defaults to 10.
        existing (dict[tuple[str,str,bool],ColorType], optional): Existing color dictionary. Defaults to dict().

    Returns:
        typing.Tuple[dict[tuple[str,str,bool],ColorType],dict[str,ColorType]]: Two dictionaries: one mapping (airport, runway, behavior) to colors, and another mapping flight id (icao24 + callsign) to colors.
    """
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
        
        if t.flight_id in id_dict.keys():
            assert id_dict[t.flight_id] == color, "Flight ID {} has multiple colors: {} and {}".format(t.flight_id,id_dict[t.flight_id],color)
        id_dict[t.flight_id] = color
    colordict_counter = i
    return runway_dict, id_dict


#################### Plotting ####################

def animate(ax:GeoAxes, airports:Airports, traffic:Traffic,
            timestamps:np.ndarray,period:int,
            speedup:float=60,
            trace_size:int=6*5, expected_speed:float=200,
            endpoints:list[FlightEndpoints]|None=None,
            colordict:dict[tuple[str,str,bool],ColorType]=dict()) -> tuple[typing.Callable[[int], list[Artist]],dict[tuple[str,str,bool],ColorType]]:
    """Animate traffic data with ``speedup`` times the source time scale.

    ``speedup=60`` means that one minute of source data takes one second to
    replay when the source sample interval is one second. The returned
    animation must be kept alive by the caller when ``show`` is false.
    """
    if speedup <= 0:
        raise ValueError("speedup must be greater than zero")
    if trace_size < 1:
        raise ValueError("trace_size must be greater than zero")

    if period <= 0:
        raise ValueError("period must be greater than zero")

    traffic:Traffic = regenerate_flight_id(traffic)
    traffic = traffic.resample(rule=f'{period}s',how='interpolate').eval(8)
    traffic = Traffic(traffic.data.dropna(subset=["latitude","longitude","flight_id","timestamp","compute_track","groundspeed"]))

    data = traffic.data
    if data.empty:
        raise ValueError("traffic contains no data")
    required_columns = {"timestamp", "latitude", "longitude"}
    missing_columns = required_columns - set(data.columns)
    if missing_columns:
        raise ValueError(f"traffic is missing columns: {sorted(missing_columns)}")

    # ax.set_xlim(data['longitude'].min(),data['longitude'].max())
    # ax.set_ylim(data['latitude'].min(),data['latitude'].max())

    flights = list(traffic)
    default_color = (0.2,0.2,0.2,0.5)
    plotnames = ('Start','End','Sim start', 'Sim end')
    markers = ('o','x','s','d')
    linestyles = ('-','-',':',':')
    shifts = (expected_speed/60,expected_speed/60,expected_speed/60,expected_speed/60)
    
    if endpoints is None:
        endpoints = generate_flightEndpoints(airports,expected_speed,expected_speed,traffic,True,None,True) 
    endpoints:list[FlightEndpoints] = regenerate_flight_id(endpoints)
    
    endpoints_dict = {e.flight_id:e for e in endpoints}
    
    colordict, id_colordict = make_colordict(endpoints,existing=colordict)

    for _, flight in enumerate(flights):
        color = id_colordict.get(str(flight.flight_id), default_color)
        if color == default_color:
            print(f"Flight {flight.flight_id} not found in colors, using default color")
        # flight.plot(ax, color=color, alpha=0.25)
        endpts = endpoints_dict.get(str(flight.flight_id),None)
        if endpts is not None:
            points = [endpts.start, endpts.end]
            for point, marker, shift in zip(points, markers,shifts):
                ax.scatter([point.longitude], [point.latitude], s=2, color=color, marker=marker,alpha=0.2,
                           transform=ccrs.PlateCarree())
                # projected = destination(point.latitude, point.longitude, point.bearing, shift * NM_TO_METERS)
                # ax.plot([point.longitude, projected[1]], [point.latitude, projected[0]], color=color, linestyle=":" if marker == "s" else "-",
                        #    transform=ccrs.PlateCarree())

    id_column = "flight_id" if "flight_id" in data.columns else "icao24"
    data["_animation_id"] = data[id_column].astype(str)
    flight_ids = data["_animation_id"].unique()
    position_lines:dict[str,PathCollection] = {}
    trace_lines:dict[str,Line2D] = {}
    forward_lines:dict[str,Line2D] = {}
    flight_labels:dict[str,Text] = {}
    for i, flight_id in enumerate(flight_ids):
        color = id_colordict.get(flight_id, None)
        if color is None:
            print(f"Flight {flight_id} not found in colors, using default color")
            color = default_color
        
        trace_lines[flight_id] = ax.plot([], [], linestyle=":", alpha=0.2, color=color, transform=ccrs.PlateCarree())[0]
        position_lines[flight_id] = ax.scatter([], [], marker="^", edgecolor=color, facecolor='none', transform=ccrs.PlateCarree())
        forward_lines[flight_id] = ax.plot([], [], linestyle="-", alpha=0.5, color=color, transform=ccrs.PlateCarree())[0]
        flight_labels[flight_id] = ax.text(np.nan, np.nan, "", ha="left", va="bottom", fontsize=8,
                                          color=color, transform=ccrs.PlateCarree())

    start_time = pd.Timestamp(timestamps[0])

    def update(frame_index:int) -> list[Artist]:
        current_time = pd.Timestamp(timestamps[frame_index])
        frame_data = data[data["timestamp"] <= current_time]
        visible_artists:list[Artist] = []
        for flight_id in flight_ids:
            flight_data = frame_data[frame_data["_animation_id"] == flight_id].tail(trace_size)
            trace = trace_lines[flight_id]
            position = position_lines[flight_id]
            label = flight_labels[flight_id]
            if flight_data.empty or flight_data["timestamp"].max() < current_time - pd.Timedelta(seconds=10*period):
                trace.set_visible(False)
                position.set_visible(False)
                label.set_visible(False)
                continue
            trace.set_data(flight_data["longitude"], flight_data["latitude"])
            latest = flight_data.iloc[-1]
            position.set_offsets([[latest["longitude"], latest["latitude"]]])
            forward = forward_lines[flight_id]
            latest_bearing:float = latest.get("compute_track", 0)
            latest_speed:float = latest.get("groundspeed", 200)/60  # Convert from knots to NM/min
            latest_forward = destination(latest["latitude"], latest["longitude"], latest_bearing, latest_speed * NM_TO_METERS)
            forward.set_data([latest["longitude"], latest_forward[1]], [latest["latitude"], latest_forward[0]])
            
            flight_endtime = traffic[flight_id].stop
            latest_callsign = str(latest.get("callsign", flight_id))
            latest_timestamp = "{:.1f} min".format((flight_endtime-current_time).total_seconds()/60)
            label.set_position((latest["longitude"], latest["latitude"]))
            label.set_text(f"{latest_callsign}:{latest_timestamp}")
            trace.set_visible(True)
            position.set_visible(True)
            label.set_visible(True)
            visible_artists.extend((trace, position, label))
        elapsed = current_time - start_time
        return visible_artists

    return update,colordict



def plot(ax:GeoAxes, airports:Airports, traffic:Traffic, expected_speed:float=200, endpoints:list[FlightEndpoints]|None=None, colordict:dict[tuple[str,str,bool],ColorType]=dict()) -> dict[tuple[str,str,bool],ColorType]:
    
    # Setup colors for plotting
    if colordict is None:
        colordict = dict()
    default_color = (0.2,0.2,0.2,0.5)
    
    traffic = regenerate_flight_id(traffic)
    
    plotnames = ('Start','End','Sim start', 'Sim end')
    markers = ('o','x','s','d')
    linestyles = ('-','-',':',':')
    shifts = (expected_speed/60,expected_speed/60,expected_speed/60,expected_speed/60)
    
    if endpoints is None:
        endpoints = generate_flightEndpoints(airports,expected_speed,expected_speed,traffic,True,None,True)    
    endpoints:list[FlightEndpoints] = regenerate_flight_id(endpoints)
    
    
    colordict, id_colordict = make_colordict(endpoints,existing=colordict)
    
        
    for flight in traffic:
        if flight.flight_id is None:
            continue
        assert isinstance(flight.flight_id,str)
    
        color = id_colordict.get(flight.flight_id, default_color)
        flight.plot(ax,color=color,alpha=0.3)
        # start       = r.start
        # end         = r.end
        # pts = [start,end]
        
        # for p,m,ls,dp,n in zip(pts,markers,linestyles,shifts,plotnames):            
        #     ax.scatter(
        #         [p.longitude],
        #         [p.latitude],
        #         s=10,color=color,marker=m,
        #         transform=ccrs.PlateCarree()
        #     )
            
        #     try:
        #         e = destination(p.latitude,p.longitude,p.bearing,dp * NM_TO_METERS)
        #     except Exception as e:
        #         print(f"Error at flight {flight} for point {n} with coordinates {p}: {e}\nSkipping this point.")
        #         continue
            
        #     ax.plot(
        #         [p.longitude, e[1]],
        #         [p.latitude, e[0]],
        #         linestyle=ls,
        #         color=color,
        #         transform=ccrs.PlateCarree()
        #     )

    
    for (a,rw,b),c in colordict.items():
        ax.plot([],[],marker='s',linestyle='',color=c,label=f"{a} ({rw} {'departing' if b else 'arriving'})")
        
    for m,ls,n in zip(markers,linestyles,plotnames):
        ax.plot([],[],marker=m,linestyle=ls,label=n,color=(0.2,0.2,0.2,0.5))
    
    
    ax.set_title("Traffic plot for {} flights, from {} to {}".format(len(traffic),
                                                                     traffic.start_time.strftime("%Y-%m-%d %H:%M"),
                                                                     traffic.end_time.strftime("%Y-%m-%d %H:%M" if traffic.start_time.date() != traffic.end_time.date() else "%H:%M")))
    
    
    min_x, max_x = ax.get_xlim()
    min_y, max_y = ax.get_ylim()
    
    xticks_vals = np.linspace(min_x, max_x, num=5)
    xticks_labels = ["{:.1f}".format(x/NM_TO_METERS) for x in xticks_vals]
    yticks_vals = np.linspace(min_y, max_y, num=5)
    yticks_labels = ["{:.1f}".format(y/NM_TO_METERS) for y in yticks_vals]


    ax.set_xticks(xticks_vals)
    ax.set_xticklabels(xticks_labels)
    ax.set_yticks(yticks_vals)
    ax.set_yticklabels(yticks_labels)
    
    return colordict
                

#################### Entrypoint ####################

__pbar = None
def main():
    import argparse
    
    parser = argparse.ArgumentParser('Traffic plotter','Traffic seen in a traffic dataset.')
    parser.add_argument('dataset',help="Path to a traffic dataset in parquet format.")
    parser.add_argument('ICAOs',nargs='+',
                            help="List of ICAO codes for airports in which to look for landings in the given dataset")
    parser.add_argument('-e','--endpoints',help='Pickle path of the companion summary files (for simulated data), containg endpoints information.',default=None)
    parser.add_argument('-f','--freq-str',dest='freq_str',default='2h',
                        help="Frequency string for grouping the data (e.g., 'D' for daily, 'W' for weekly). Full details at https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases")
    parser.add_argument('-a','--animate',action='store_true',help="Animate the traffic data instead of plotting it.")
    parser.add_argument('--twin',default=None,help="Path to a twin traffic dataset in parquet format, to be plotted alongside the main dataset.")
    parser.add_argument('--twin-ends',dest='twin_ends',default=None,help="Path to a twin summary file in pickle format, to be used alongside the main summary file.")
    parser.add_argument('--save',dest='save',default=None,help="Path to a folder for saving the generated plot or animation. If not provided, the plot will be shown interactively.")
    
    args = parser.parse_args()
    
    
    print("ICAO codes to look for landings:",args.ICAOs)
    airports = get_airports(args.ICAOs)
    transformer = get_airport_latlon_transformer(airports[0])
    projection = ccrs.Projection(transformer.target_crs)
    
    speedup = 100.0
    period = 10 # seconds
    interval_ms = max(1, period * 1000 / speedup)
    
    dataset_path = pathlib.Path(args.dataset)
    if not(dataset_path.is_file()):
        print("ERROR: Dataset path is not a file")
        exit(1)
    traffic_gen = iter_flight_by_period(args.dataset, freq_str=args.freq_str)
    
    if args.endpoints is not None:
        with open(args.endpoints, 'rb') as f:
            summary = pickle.load(f)
            assert isinstance(summary, SimulationSummary), f"The summary file {args.endpoints} is not a SimulationSummary"
            endpoints = summary.tasks
    else:
        endpoints = None
        
    if args.twin is not None:
        twin_path = pathlib.Path(args.twin)
        if not(twin_path.is_file()):
            print("ERROR: Twin dataset path is not a file")
            exit(1)
        twin_gen = iter_flight_by_period(args.twin, freq_str=args.freq_str)
        if args.twin_ends is not None:
            with open(args.twin_ends, 'rb') as f:
                twin_summary = pickle.load(f)
                assert isinstance(twin_summary, SimulationSummary), f"The twin summary file {args.twin_ends} is not a SimulationSummary"
                twin_endpoints = twin_summary.tasks
        else:
            twin_endpoints = None
    else:
        twin_gen = itertools.repeat(None)
        twin_endpoints = None
    
    colordict = dict()
    for traffic,twin in zip(traffic_gen,twin_gen):
        print(f"Processing traffic from {traffic.data['timestamp'].min()} to {traffic.data['timestamp'].max()}")
        if twin is not None:
            fig,axes = plt.subplots(1,2,figsize=(16,8),sharex=True,sharey=True,subplot_kw={'projection': projection}) # type: ignore
            ax:GeoAxes = axes[0]
            twin_ax:GeoAxes = axes[1]
        else:
            fig,ax = plt.subplots(1,1,figsize=(8,8),subplot_kw={'projection': projection}) # type: ignore
            twin_ax = None
        filtered_traffic = filter_traffic(traffic,airports)
        if filtered_traffic is None:
            print("ERROR: Could not filter traffic over the given airports, skipping this period")
            continue
        
        if twin is not None:
            filtered_twin = filter_traffic(twin,airports)
            if filtered_twin is None:
                print("ERROR: Could not filter twin traffic over the given airports, skipping this period")
                continue
        else:
            filtered_twin = None
        
        if args.animate:
            min_time = min(traffic.start_time, twin.start_time) if twin is not None else traffic.start_time
            max_time = max(traffic.end_time, twin.end_time) if twin is not None else traffic.end_time
            timestamps = pd.date_range(start=min_time, end=max_time, freq=f'{period}s').to_numpy()
            
            update_fun, colordict = animate(ax, airports, filtered_traffic, timestamps, period, endpoints=endpoints, colordict=colordict)
            if filtered_twin is not None:
                twin_update_fun, colordict = animate(twin_ax, airports, filtered_twin, timestamps, period, endpoints=twin_endpoints, colordict=colordict)
                def combined_update(frame_index:int) -> list[Artist]:
                    artists1 = update_fun(frame_index)
                    artists2 = twin_update_fun(frame_index)
                    fig.suptitle(f"Traffic replay: {timestamps[frame_index].strftime('%H:%M:%S')}")
                    return artists1 + artists2
                animation = FuncAnimation(fig, combined_update, frames=len(timestamps), interval=interval_ms, blit=False, repeat=True)
            else:
                def update_fun_with_title(frame_index:int) -> list[Artist]:
                    artists = update_fun(frame_index)
                    fig.suptitle(f"Traffic replay: {timestamps[frame_index].strftime('%H:%M:%S')}")
                    return artists
                animation = FuncAnimation(fig, update_fun_with_title, frames=len(timestamps), interval=interval_ms, blit=False, repeat=True)
            
            for (a,rw,b),c in colordict.items():
                ax.plot([],[],marker='s',linestyle='',color=c,label=f"{a} ({rw} {'departing' if b else 'arriving'})")
        else:
            animation = None
            colordict = plot(ax,airports, filtered_traffic, endpoints=endpoints, colordict=colordict)
            if filtered_twin is not None:
                colordict = plot(twin_ax,airports, filtered_twin, endpoints=twin_endpoints, colordict=colordict)
                
        ax.set_xlabel(f"Easting (NM, {transformer.target_crs.name})")
        ax.set_ylabel("Northing (NM)")
        ax.set_aspect('equal')
        ax.legend()
        if twin_ax is not None:
            xlimits = ax.get_xlim()
            ylimits = ax.get_ylim()
            
            xlimits_bis = twin_ax.get_xlim()
            ylimits_bis = twin_ax.get_ylim()
            
            xlims = (min(xlimits[0], xlimits_bis[0]), max(xlimits[1], xlimits_bis[1]))
            ylims = (min(ylimits[0], ylimits_bis[0]), max(ylimits[1], ylimits_bis[1]))
            
            ax.set_xlim(xlims)
            ax.set_ylim(ylims)
            twin_ax.set_xlim(xlims)
            twin_ax.set_ylim(ylims)
            
            twin_ax.set_xlabel(f"Easting (NM, {transformer.target_crs.name})")
            twin_ax.set_ylabel("Northing (NM)")
            twin_ax.set_aspect('equal')
            twin_ax.legend()
        
        
        fig.tight_layout()

        if args.save is not None:
            save_path = pathlib.Path(args.save)
            os.makedirs(save_path,exist_ok=True)
            if animation is not None:
                animation_save_path = save_path / f"traffic_animation_{traffic.start_time.strftime('%Y-%m-%d_%Hh%Mm%Ss')}_{traffic.end_time.strftime('%Y-%m-%d_%Hh%Mm%Ss')}.mp4"
                def tqdm_callback(frame_i:int,frame_total:int):
                    global __pbar
                    if __pbar is None:
                        __pbar = tqdm.tqdm(total=frame_total,
                                         desc=f"Saving animation to {animation_save_path}")
                    __pbar.update(1)
                
                animation.save(animation_save_path, writer='ffmpeg', fps=6, progress_callback=tqdm_callback)
                if __pbar is not None:
                    __pbar.close()
            else:
                plot_save_path = save_path / f"traffic_plot_{traffic.start_time.strftime('%Y-%m-%d_%Hh%Mm%Ss')}_{traffic.end_time.strftime('%Y-%m-%d_%Hh%Mm%Ss')}.svg"
                print(f"Saving plot to {plot_save_path}")
                fig.savefig(plot_save_path)
        else:
            plt.show()

        cont = input("Enter 'y' to start simulation with the next period, quit otherwise: ")
        if cont != 'y':
            break


if __name__ == '__main__':
    main()
