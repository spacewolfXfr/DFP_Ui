
#!/usr/bin/python3
import pickle,pathlib

import numpy as np

import pandas as pd

import dataclasses
from argparse import Namespace

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from .FlightExtraction import FlightEndpoints


#################### Simulation summary ####################

@dataclasses.dataclass
class SimulationSummary:
    tasks:list[FlightEndpoints]
    command_args:Namespace

def count_aircraft_in_flight(tasks:list[FlightEndpoints], time_window:pd.Timedelta, time_step:pd.Timedelta) -> tuple[list[pd.Timestamp],list[int]]:
    tasks.sort(key=lambda t : t.end_time)
    
    curr_t = min(t.start_time for t in tasks)
    end_t = max(t.end_time for t in tasks)
    
    
    times = []
    counts = []
    
    while curr_t < end_t:
        count = 0
        window_start = curr_t - time_window
        for t in tasks:
            if t.start_time < window_start:
                continue
            else:
                if t.end_time < curr_t:
                    count+=1
                else: # Sorted by end time
                    break
        
        times.append(curr_t)
        counts.append(count)
        curr_t += time_step
        
    return times,counts


def plot_endpoints_results(tasks:list[FlightEndpoints]):
    arriving_dts = []
    departing_dts = []
    arriving_ends = []
    departing_ends = []
    removed_arriving_time = []
    removed_departing_time = []
    removed_tasks_count = 0
    pending_tasks_count = 0
    for task in tasks:
        # Consider only tasks ended and not removed due to loss of separation
        if task.removed:
            removed_tasks_count += 1
            if task.departing:
                removed_departing_time.append(task.end_time)
            else:
                removed_arriving_time.append(task.end_time)
        elif task.completed:
            dt = (task.end_time - task.initial_end_time).total_seconds()/60
            if task.departing:
                departing_dts.append(dt)
                departing_ends.append(task.end_time)
            else:
                arriving_dts.append(dt)
                arriving_ends.append(task.end_time)
        else:
            pending_tasks_count += 1
            
    dts = np.array(arriving_dts+departing_dts)
    ends = np.array(arriving_ends+departing_ends)
    
    print(f"Number of pending tasks: {pending_tasks_count}")
    
    if len(dts) == 0 or len(ends) == 0:
        print("WARNING: No data collected, cannot produce report")
        return
    
    min_dt = int(np.floor(np.min(dts)))
    max_dt = int(np.ceil(np.max(dts)))
    abs_max_dt = max(abs(min_dt),abs(max_dt))
    
    minrange = -30 if min_dt-1 < -30 else min_dt
    maxrange = 30 if max_dt+1 > 30 else max_dt
    edges = [min_dt-1] + list(range(minrange,maxrange,2)) + [max_dt+1]
    
    hist,edges = np.histogram(dts,edges)
    mean = np.mean(dts)
    median = np.median(dts)
    std = np.std(dts)
    
    
    fig,axes = plt.subplots(2,1)
    axes:tuple[Axes,Axes,Axes]
    bars = axes[0].bar(edges[:-1], hist, width=np.diff(edges), align='edge',edgecolor='k')
    axes[0].bar_label(bars, fontsize=20, color='navy')
    axes[0].set_xlabel('Delay (minutes)')
    axes[0].set_xticks(edges)
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Distribution of Flight Delays ({} flights considered ; {} removed)'.format(len(dts), removed_tasks_count))
    axes[0].vlines(mean, ymin=0, ymax=max(hist), colors='k', linestyles='dashed', label='Mean: {:.2f} min'.format(mean))
    axes[0].vlines(median, ymin=0, ymax=max(hist), colors='k', linestyles='solid', label='Median: {:.2f} min'.format(median))
    axes[0].vlines([mean + std, mean - std], ymin=0, ymax=max(hist), colors='g', linestyles='dotted', label='Mean $\\pm$ 1 Std Dev: {:.2f} min'.format(std))
    axes[0].legend()
    
    
    # Lollipop plot
    cmap = plt.get_cmap('coolwarm')
    colors = cmap((1+dts/abs_max_dt)/2)
    
    arriving_colors = cmap((1+np.array(arriving_dts)/abs_max_dt)/2)
    departing_colors = cmap((1+np.array(departing_dts)/abs_max_dt)/2)
    
    axes[1].vlines(ends, 0, dts, color=colors, alpha=0.7,zorder=0)
    axes[1].scatter(arriving_ends,arriving_dts, marker='o', c=arriving_colors, edgecolors='k', label='Arriving',zorder=2.2)
    axes[1].scatter(departing_ends,departing_dts, marker='s', c=departing_colors, edgecolors='k', label='Departing',zorder=2.2)
    axes[1].scatter(removed_arriving_time,np.zeros(len(removed_arriving_time)), marker='o', c='k', edgecolors='k', label='Removed (Arriving)', zorder=2.2)
    axes[1].scatter(removed_departing_time,np.zeros(len(removed_departing_time)), marker='s', c='k', edgecolors='k', label='Removed (Departing)', zorder=2.2)
    
    xlims = axes[1].get_xlim()
    
    # Aircraft count plot
    twinax = axes[1].twinx()
    times,counts = count_aircraft_in_flight(tasks,pd.Timedelta(minutes=60),pd.Timedelta(minutes=1))
    twinax.step(times,counts,color='k',where='pre',zorder=1.5,label='Aircraft in flight')
    twinax.fill_between(times,counts,step='pre',color='gray',alpha=0.2,zorder=1)
    twinax.set_xlim(xlims)
    twinax.set_ylim(0,max(counts)+1)
    twinax.set_ylabel('Aircraft count')
    twinax.yaxis.get_major_locator().set_params(integer=True)
    twinax.grid(True,axis='y',linestyle=':')
    twinax.legend()
    
    axes[1].hlines(0,xlims[0],xlims[1],'k',zorder=0.5)
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Delay (minutes)')
    axes[1].set_title('Flight Delays Over Time')
    axes[1].grid(True,axis='y')
    axes[1].legend()
            
    plt.show()
    
    
def main():
    import argparse
    
    parser = argparse.ArgumentParser('Simulation summary visualiser')
    parser.add_argument('input_file', type=pathlib.Path, help='Path to the pickle file containing the list of FlightEndpoints tasks')
    args = parser.parse_args()
    
    with open(args.input_file, 'rb') as f:
        summary = pickle.load(f)
        
    assert isinstance(summary, SimulationSummary), "The input file does not contain a SimulationSummary object"
    print("Arguments used for the simulation:\n {}".format(summary.command_args))
    plot_endpoints_results(summary.tasks)
    
if __name__ == '__main__':
    main()