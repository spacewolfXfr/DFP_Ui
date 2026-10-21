
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


def mean_med_std(arr:np.ndarray) -> tuple[float,float,float]:
    mean = np.mean(arr)
    median = np.median(arr)
    std = np.std(arr)
    return mean,median,std

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
    arriving_dts = np.array(arriving_dts)
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
    arriving_hist,arriving_edges = np.histogram(arriving_dts,edges)
    
    arr_mean, arr_median, arr_std = mean_med_std(np.array(arriving_dts))
    dep_mean, dep_median, dep_std = mean_med_std(np.array(departing_dts))    
        
    fig_hist,hist_ax = plt.subplots(1,1,figsize=(16,9))
    bars = hist_ax.bar(edges[:-1], hist, width=np.diff(edges), align='edge',edgecolor='k',facecolor='lightblue',alpha=0.8)
    hist_ax.bar_label(bars, fontsize=16, color='k')
    
    landing_bars = hist_ax.bar(arriving_edges[:-1], arriving_hist, width=np.diff(arriving_edges), align='edge', edgecolor='w', facecolor='darkblue')
    
    props = arriving_hist/hist
    prop_labels = []
    for p in props:
        if np.isnan(p) or p > 0.99999:
            prop_labels.append('')
        else:
            prop_labels.append('{:.0f}%'.format(p*100))
    
    
    hist_ax.bar_label(landing_bars,
                      labels = prop_labels,
                      fontsize=12, color='white',
                      label_type='center')
    
    hist_ax.set_xlabel('Delay (minutes)')
    hist_ax.set_xticks(edges)
    hist_ax.set_ylabel('Number of flights')
    hist_ax.set_title('Distribution of Flight Delays ({} flights considered ; {} removed)'.format(len(dts), removed_tasks_count))
    
    hist_ax.vlines(arr_mean, ymin=0, ymax=max(hist), colors='navy', linestyles='dashed', label='Arriving Mean: {:.2f} min'.format(arr_mean))
    hist_ax.vlines(arr_median, ymin=0, ymax=max(hist), colors='navy', linestyles='solid', label='Arriving Median: {:.2f} min'.format(arr_median))
    hist_ax.vlines([arr_mean + arr_std, arr_mean - arr_std], ymin=0, ymax=max(hist), colors='g', linestyles='dotted', label='Arriving Std Dev: {:.2f} min'.format(arr_std))
    
    hist_ax.vlines(dep_mean, ymin=0, ymax=max(hist), colors='gray', linestyles='dashed', label='Departing Mean: {:.2f} min'.format(dep_mean))
    hist_ax.vlines(dep_median, ymin=0, ymax=max(hist), colors='gray', linestyles='solid', label='Departing Median: {:.2f} min'.format(dep_median))
    hist_ax.vlines([dep_mean + dep_std, dep_mean - dep_std], ymin=0, ymax=max(hist), colors='r', linestyles='dotted', label='Departing Std Dev: {:.2f} min'.format(dep_std))
    
    hist_ax.set_xlim(min_dt-1,max_dt+1)
    hist_ax.grid(True,axis='y',linestyle=':')
    hist_ax.legend()
    fig_hist.tight_layout()
    
    
    min_datetime:pd.Timestamp = min(arriving_ends+departing_ends)
    max_datetime:pd.Timestamp = max(arriving_ends+departing_ends)
        
    min_datetime = min_datetime.floor('30min')
    max_datetime = max_datetime.ceil('30min')
    
    xlims = (min_datetime,max_datetime)
    
    # Lollipop plot
    fig_time,time_ax = plt.subplots(1,1,figsize=(16,9))
    cmap = plt.get_cmap('coolwarm')
    colors = cmap((1+dts/abs_max_dt)/2)
    
    arriving_colors = cmap((1+np.array(arriving_dts)/abs_max_dt)/2)
    departing_colors = cmap((1+np.array(departing_dts)/abs_max_dt)/2)
    
    time_ax.vlines(ends, 0, dts, color=colors, alpha=0.7,zorder=0)
    time_ax.scatter(arriving_ends,arriving_dts, marker='o', c=arriving_colors, edgecolors='k', label='Arriving',zorder=2.2)
    time_ax.scatter(departing_ends,departing_dts, marker='s', c=departing_colors, edgecolors='k', label='Departing',zorder=2.2)
    if len(removed_arriving_time) > 0:
        time_ax.scatter(removed_arriving_time,np.zeros(len(removed_arriving_time)), s=100, marker='o', c='k', edgecolors='r', label='Removed (Arriving)', zorder=2.2)
    if len(removed_departing_time) > 0:
        time_ax.scatter(removed_departing_time,np.zeros(len(removed_departing_time)), s=100, marker='s', c='k', edgecolors='r', label='Removed (Departing)', zorder=2.2)

    time_ax.set_xlim(xlims)
    
    # Aircraft count plot
    twinax = time_ax.twinx()
    times,counts = count_aircraft_in_flight(tasks,pd.Timedelta(minutes=60),pd.Timedelta(minutes=1))
    twinax.step(times,counts,color='gray',alpha=0.7,where='pre',zorder=1.5,label='Aircraft in flight')
    twinax.fill_between(times,counts,step='pre',color='gray',alpha=0.2,zorder=1)
    twinax.set_xlim(xlims)
    twinax.set_ylim(0,max(counts)+1)
    twinax.set_ylabel('Aircraft count')
    twinax.yaxis.get_major_locator().set_params(integer=True)
    twinax.grid(True,axis='y',linestyle=':')
    twinax.legend().set_loc('upper right')
        
    ticks = pd.date_range(min_datetime, max_datetime, freq='h')
    labels = ticks.strftime('%H:%M')
    
    time_ax.hlines(0,xlims[0],xlims[1],'k',zorder=0.5)
    time_ax.set_xticks(ticks,labels)
    time_ax.set_xlabel('Time of arrival')
    time_ax.set_ylabel('Delay (minutes)')
    title = 'Flight Delays over time'
    if min_datetime.date() == max_datetime.date():
        title += ' for {}'.format(min_datetime.strftime('%Y-%m-%d'))
    else:
        title += ' from {} to {}'.format(min_datetime.strftime('%Y-%m-%d'),max_datetime.strftime('%Y-%m-%d'))
    time_ax.set_title(title)
    time_ax.grid(True,axis='y')
    time_ax.legend().set_loc('upper left')
    fig_time.tight_layout()
    
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