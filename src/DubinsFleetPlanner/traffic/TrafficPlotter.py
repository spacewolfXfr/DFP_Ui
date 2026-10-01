#!/usr/bin/python3

from __future__ import annotations

import pathlib
import typing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import colormaps as cm
from matplotlib.axes import Axes


Colortype = tuple[float,float,float,float]

import pandas as pd

from pitot.geodesy import destination

from traffic.core import Traffic
from traffic.data import samples

from cartopy import crs as ccrs

from DubinsFleetPlanner.ioUtils import ACStats
from DubinsFleetPlanner.thread_estimator import thread_count

from .FlightExtraction import extract_flight_endpoints,NM_TO_METERS
from .TrafficReader import filter_traffic,iter_flightdata_by_day
from .AirportHelpers import get_airports,Airports


#################### Plotting ####################

def plot(airports:Airports, traffic:Traffic, expected_speed:float=200, threshold_shift:float=20):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Setup colors for plotting
    cmap = cm["tab10"]
    cmap_i = 0
    cmap_div = 10
    colordict:dict[tuple[str,str],Colortype] = dict()
    leaving_color = (0.2,0.2,0.2,0.5)
    
    plotnames = ('Start','End','Estimated IF')
    markers = ('o','x','s')
    linestyles = ('-','-',':')
    shifts = (expected_speed/60,expected_speed/60,threshold_shift)
    
    for i,flight in enumerate(traffic):
        stats = ACStats(
            i,
            expected_speed/60, # Convert from kts (NM/h) to NM/minute
            1.,
            expected_speed/(60*np.pi) # Full circle in 2 minutes
        )
        
        r = extract_flight_endpoints(flight,airports,stats,threshold_shift)
        
        if r is None:
            flight.plot(ax,color=leaving_color,alpha=0.3)
            continue
        else:
            if r[0].dest_ICAO is not None and r[0].dest_runway is not None:
                if (r[0].dest_ICAO,r[0].dest_runway) not in colordict.keys():
                    colordict[(r[0].dest_ICAO,r[0].dest_runway)] = cmap(cmap_i/cmap_div)
                    cmap_i = (cmap_i+1) % cmap_div
            elif r[0].start_ICAO is not None and r[0].start_runway is not None:
                if (r[0].start_ICAO,r[0].start_runway) not in colordict.keys():
                    colordict[(r[0].start_ICAO,r[0].start_runway)] = cmap(cmap_i/cmap_div)
                    cmap_i = (cmap_i+1) % cmap_div
                color = colordict[(r[0].start_ICAO,r[0].start_runway)]
            else:
                color = leaving_color
            
            flight.plot(ax,color=color,alpha=0.3)
            start       = r[0].start
            end         = r[0].end
            pts = [start,end]
            if r[1] is not None:
                pts.append(r[1].end)
            
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
    
    
    
    for (a,rw),c in colordict.items():
        ax.plot([],[],marker='s',linestyle='',color=c,label=f"{a} ({rw})")
        
    for m,ls,n in zip(markers,linestyles,plotnames):
        ax.plot([],[],marker=m,linestyle=ls,label=n,color=(0.2,0.2,0.2,0.5))
    
    ax.legend()
    fig.tight_layout()
    plt.show()
                

#################### Entrypoint ####################

def main():
    import argparse
    
    parser = argparse.ArgumentParser('Traffic plotter','Traffic seen in a traffic dataset.')
    parser.add_argument('--demo',action='store_true',help="Run a demo with a sample dataset.")
    parser.add_argument('dataset',help="Path to a traffic dataset in parquet format.")
    parser.add_argument('ICAOs',nargs='+',
                            help="List of ICAO codes for airports in which to look for landings in the given dataset")
    
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
        plot(airports, filtered_traffic)
        exit(0)
    
    dataset_path = pathlib.Path(args.dataset)
    if not(dataset_path.is_file()):
        print("ERROR: Dataset path is not a file")
        exit(1)
    traffic_gen = iter_flightdata_by_day(args.dataset)
    for traffic in traffic_gen:
        print(f"Processing traffic for day {traffic.data['timestamp'].min().date()}")
        filtered_traffic = filter_traffic(traffic,airports)
        if filtered_traffic is None:
            print("ERROR: Could not filter the day of traffic")
            continue
        plot(airports, filtered_traffic)


if __name__ == '__main__':
    main()
