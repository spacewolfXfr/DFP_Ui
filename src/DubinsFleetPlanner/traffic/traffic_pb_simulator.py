#!/usr/bin/python3

from __future__ import annotations

import dataclasses
import typing
from typing import Optional
import subprocess
import pathlib
import json

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from matplotlib.patches import Circle
from matplotlib import colormaps as cm

Colortype = tuple[float,float,float,float]

import pandas as pd

from traffic.core import Traffic
from traffic.data import samples
from traffic.algorithms import filters


from DubinsFleetPlanner.ioUtils import AC_PP_Problem,ACStats,write_pathplanning_problem_to_CSV,print_FleetPlan_to_JSON,parse_trajectories_from_JSON
from DubinsFleetPlanner.Dubins import Path,BasicPath,FleetPlan,DubinsMove
from DubinsFleetPlanner.Poses import Pose2D,Pose3D,min_XY_dist

from pyproj import Transformer

from pitot.geodesy import distance


from .AirportHelpers import get_airport, get_airports, get_airport_latlon_transformer
from .FlightExtraction import FlightEndpoints,LatlonPose,extract_flight_endpoints,flight_landing,NM_TO_METERS,get_other_runway_name,get_runway
from .TrafficReader import iter_flightdata_by_day,filter_traffic

from DubinsFleetPlanner.UI.plotting import plot_pose2d_sequence,transpose_list_of_trajectories

#################### Utility ####################

def make_box_boundary_file(xmin:float,xmax:float,ymin:float,ymax:float,filepath:str|pathlib.Path):
        low_left    = (xmin, ymin, 0)
        low_right   = (xmax, ymin, 0)
        high_left   = (xmin, ymax, 0)
        high_right  = (xmax, ymax, 0)
        
        sections = [
            BasicPath.from_2_points(low_left,low_right).asdict(),
            BasicPath.from_2_points(low_left,high_left).asdict(),
            BasicPath.from_2_points(high_right,low_right).asdict(),
            BasicPath.from_2_points(high_left,high_right).asdict()
        ]
        
        with open(filepath,mode='w') as f:
            json.dump({'sections':sections},f)
            
def make_circle_boundary_file(xc:float,yc:float,radius:float,filepath:str|pathlib.Path):
        sections = [
            BasicPath.from_circle(xc,yc,radius).asdict()
        ]
        
        with open(filepath,mode='w') as f:
            json.dump({'sections':sections},f)
            

@dataclasses.dataclass
class InfluenceCircle:
    lat:float
    lon:float
    radius:float # In NM
    
    def to_xy_meters(self,transformer:Transformer) -> tuple[float,float,float]:
        x,y = transformer.transform(self.lat,self.lon)
        return (x,y,self.radius*NM_TO_METERS)
    
    def to_xy_NM(self,transformer:Transformer) -> tuple[float,float,float]:
        x,y = transformer.transform(self.lat,self.lon)
        return (x/NM_TO_METERS,y/NM_TO_METERS,self.radius)
    
    @staticmethod
    def from_xy_meters(x:float,y:float,radius_meters:float,transformer:Transformer) -> InfluenceCircle:
        lat,lon = transformer.transform(x,y,direction='INVERSE')
        return InfluenceCircle(lat,lon,radius_meters/NM_TO_METERS)

#################### Ongoing flights simulator ####################


def solve_problem(solver:pathlib.Path,src_dir:pathlib.Path,dest_dir:pathlib.Path,
                   separation:float, wind:tuple[float,float],threads:int,
                   obstacle_path:typing.Optional[pathlib.Path],
                   geometric_obstacles_path:typing.Optional[pathlib.Path],
                   start_extensions:list[float] = [],
                   end_extensions:list[float] = []) -> subprocess.CompletedProcess[bytes]:
    cmd = []
    cmd.append(str(solver.resolve()))
    cmd.append(str(src_dir.resolve()))
    cmd.append(str(dest_dir.resolve()))
    cmd.append(str(separation))
    
    cmd.append(str(wind[0]))
    cmd.append(str(wind[1]))
    
    cmd.append('-t')
    cmd.append(str(threads))
    
    cmd.append('-l')
    
    cmd.append('--samples')
    cmd.append('3')
    
    cmd.append('--ellipse')
    cmd.append('0.4')
    
    cmd.append('--ellipse-border-only')
    
    cmd.append('-v')
    cmd.append('2')
    
    cmd.append('--straights-only')
    # cmd.append('--allow-shortest')
    
    if obstacle_path is not None:
        cmd.append('-O')
        cmd.append(obstacle_path.resolve())
        
    if geometric_obstacles_path is not None:
        cmd.append('-G')
        cmd.append(geometric_obstacles_path.resolve())
        
    if len(start_extensions) > 0:
        cmd.append('--extend-start')
        for e in start_extensions:
            cmd.append(str(e))
    
    if len(end_extensions) > 0:
        cmd.append('--extend-end')
        for e in end_extensions:
            cmd.append(str(e))
    
    # print(cmd)
    output = subprocess.run(cmd)
    # print(output)
    output.check_returncode()
    return output

@dataclasses.dataclass
class ArrivalsSimulator:
    solver_path:pathlib.Path
    transformer:Transformer  # Convert (lat,lon) coordinates from WGS84 into some (x,y) coordinates for the solver. Must be a Conformal projection (preserve angles)
    timeshifts:list[pd.Timedelta]
    threshold_shift:float   # By how much the endpoint have been moved along a straight line, in NM
    cmd_shift:float         # Minimal straight length at the start of newly planned trajectory, in NM
    tasklist:list[FlightEndpoints]
    obstacles_json_path:pathlib.Path            = pathlib.Path("obstacles_json.json")
    output_json_path:pathlib.Path               = pathlib.Path("output_json.json")
    input_csv_path:pathlib.Path                 = pathlib.Path("input_csv.csv")
    geometric_obstacle_json_path:Optional[pathlib.Path]   = None
    influence_circle:Optional[InfluenceCircle]  = None # Circle of influence for the solver, in (lat,lon,radius) in NM. If None, no influence circle is used.
    __xy_circle:Optional[tuple[float,float,float]] = dataclasses.field(default=None,init=False) # Circle of influence for the solver, in (x,y,radius) in NM. If None, no influence circle is used.
    max_reschedule:int  = 3  # Maximum number of reschedule for each aircraft
    separation:float    = 5. # Overriden by the values in `flying` if it is defined
    z_alpha:float       = 1. # Overriden by the values in `flying` if it is defined
    wind_x:float        = 0. # Overriden by the values in `flying` if it is defined
    wind_y:float        = 0. # Overriden by the values in `flying` if it is defined
    scheduled:typing.Optional[FleetPlan] = None
    schedule_counters:dict[int,int]   = dataclasses.field(init=False) # Dict from AC id to number of reschedulings
    line_buffersize:int = 20 # Number of past points kept in visualisation
    
    __last_global_schedule:pd.Timestamp     = dataclasses.field(init=False)
    __last_schedules:dict[int,pd.Timestamp] = dataclasses.field(init=False)
    __task_index:dict[int,int]    = dataclasses.field(init=False) # Reverse accessor from Aircraft ID to tasklist index
    __t:pd.Timestamp              = dataclasses.field(init=False)
    __end_of_times:pd.Timestamp   = dataclasses.field(init=False)
    __encountered_exception:typing.Optional[Exception] = dataclasses.field(default=None,init=False)
    
    __axes:typing.Optional[Axes]    = dataclasses.field(default=None,init=False)
    __line_dict:dict[int,Line2D]    = dataclasses.field(default_factory=dict,init=False)
    __pos_dict:dict[int,Line2D]     = dataclasses.field(default_factory=dict,init=False)
    __traj_dict:dict[int,Line2D]    = dataclasses.field(default_factory=dict,init=False)
    __quiver_dict:dict[int,Quiver]  = dataclasses.field(default_factory=dict,init=False)
    __dest_dict:dict[tuple[str,str],Line2D]    = dataclasses.field(default_factory=dict,init=False)
    __color_dict:dict[int,Colortype]= dataclasses.field(default_factory=dict,init=False)
    __label_dict:dict[int,Text]     = dataclasses.field(default_factory=dict,init=False)
    __min_dist_line:typing.Optional[Line2D] = dataclasses.field(default=None,init=False)
    
    def __post_init__(self):
        self.schedule_counters = dict()
        self.tasklist.sort(key=lambda fpts : fpts.start_time)
        
        self.__t = self.tasklist[0].start_time
        self.__last_global_schedule = self.__t
        self.__end_of_times = max(t.end_time for t in self.tasklist)
        self.__task_index = dict()
        self.__encountered_exception = None
        self.__last_schedules = dict()
        
        self.__axes = None
        
        for i,t in enumerate(self.tasklist):
            self.__task_index[t.stats.id] = i
            self.schedule_counters[t.stats.id] = 0
        
        if self.scheduled is not None:
            self.separation = self.scheduled.separation
            self.z_alpha = self.scheduled.z_alpha
            self.wind_x = self.scheduled.wind_x
            self.wind_y = self.scheduled.wind_y
    
    def attach_axes(self,ax:Axes,set_xylims:bool=True):
        self.__axes = ax
        self.__min_dist_line = ax.plot([],[],marker='D',linestyle=':',label=f"Min distance (-,-) : - NM",
                                       markeredgecolor='r',color='k')[0]
        
        # Setting up colors by destination
        icao_rw_colordict = dict()
        my_cm = cm["tab10"]
        i = 0
        for t in self.tasklist:
            try:
                color = icao_rw_colordict[(t.dest_ICAO,t.dest_runway)]
            except KeyError:
                color = my_cm(i/10)
                i+=1
                icao_rw_colordict[(t.dest_ICAO,t.dest_runway)] = color
            self.__color_dict[t.stats.id] = color
                
        
        if set_xylims:
            maxx = -np.inf
            maxy = -np.inf
            minx = np.inf
            miny = np.inf
            
            for t in self.tasklist:
                start_p = t.start.to_pose3D(self.transformer,True)
                end_p = t.end.to_pose3D(self.transformer,True)
                
                maxx = max(maxx,start_p.x)
                maxx = max(maxx,end_p.x)
                
                maxy = max(maxy,start_p.y)
                maxy = max(maxy,end_p.y)
                
                minx = min(minx,start_p.x)
                minx = min(minx,end_p.x)
                
                miny = min(miny,start_p.y)
                miny = min(miny,end_p.y)
                
                end_proj = t.end.project(self.threshold_shift*NM_TO_METERS).to_pose3D(self.transformer,True)
                
                color = self.__color_dict[t.id]
                if t.dest_airport is not None and t.dest_ICAO is not None and t.dest_runway is not None:
                    if (t.dest_ICAO,t.dest_runway) not in self.__dest_dict:
                        
                        rw_1 = get_runway(t.dest_airport,t.dest_runway)
                        rw_2 = get_runway(t.dest_airport,get_other_runway_name(t.dest_runway))
                        
                        assert rw_1 is not None
                        assert rw_2 is not None
                        
                        proj1 = self.transformer.transform(rw_1.latitude,rw_1.longitude)
                        proj2 = self.transformer.transform(rw_2.latitude,rw_2.longitude)
                        
                        dest_line = self.__axes.plot(
                            [proj1[0]/NM_TO_METERS,proj2[0]/NM_TO_METERS],[proj1[1]/NM_TO_METERS,proj2[1]/NM_TO_METERS],
                            linestyle='-',
                            alpha=0.5,
                            color=color,
                            label=f"{t.dest_ICAO} : {t.dest_runway}"
                        )[0]
                        
                        self.__dest_dict[(t.dest_ICAO,t.dest_runway)] = dest_line
                    
            self.__axes.set_xlim(minx,maxx)
            self.__axes.set_ylim(miny,maxy)
            
            self.__axes.set_xlabel(f"Easting (NM, {self.transformer.target_crs.name})")
            self.__axes.set_ylabel(f"Northing (NM, {self.transformer.target_crs.name})")
            
            self.__axes.legend()
            
        
    
    def get_task(self,ac_id:int) -> FlightEndpoints:
        return self.tasklist[self.__task_index[ac_id]]
    
    def is_running(self,ac_id:int) -> bool:
        task = self.get_task(ac_id)
        return task.start_time <= self.__t and task.end_time > self.__t
    
    def is_scheduled(self,ac_id:int) -> bool:
        if self.scheduled is None:
            return False
        else:
            return ac_id in self.scheduled._traj_dict.keys()
    
    
    
    def step(self,timedelta:pd.Timedelta,
             reschedule_threshold:pd.Timedelta,final_time:pd.Timedelta,
             reschedule:bool=False,threads:int=0) -> list[tuple[ACStats,Pose3D]]:
        output:list[tuple[ACStats,Pose3D]] = []
        
        ### Time forward
        new_t = timedelta + self.__t
        
        ### Metadata
        if self.influence_circle is not None and self.__xy_circle is None:
            self.__xy_circle = self.influence_circle.to_xy_NM(self.transformer)
        
        ## Move forward and gather candidates
        
        candidate_acs:set[int] = set()
        added_acs:set[int] = set()
        if self.scheduled is not None:
            self.scheduled = self.scheduled.follow_for(timedelta.total_seconds()/60)
            for s,p in self.scheduled.trajectories:
                output.append((s,p.start))
                added_acs.add(s.id)
                
                add_me = False
                # Reschedule only possible during a straight without incoming turn
                if p.sections[0].type == DubinsMove.STRAIGHT:
                    if len(p.junctions) > 0:
                        if p.junctions[0] > reschedule_threshold.total_seconds()/60:
                            add_me = True
                    else:
                        if p.duration() > reschedule_threshold.total_seconds()/60:
                            add_me = True
                
                # Don't reschedule if outside the influence circle
                if self.__xy_circle is not None:
                    dx = p.start.x - self.__xy_circle[0]
                    dy = p.start.y - self.__xy_circle[1]
                    if dx*dx + dy*dy > self.__xy_circle[2]*self.__xy_circle[2]:
                        add_me = False
                
                if add_me:
                    candidate_acs.add(s.id)
        
              
        for i,t in enumerate(self.tasklist):
            id = t.stats.id
            
            # Task already ended: remove drawing
            if t.end_time <= new_t:
                if (new_t - t.end_time) > timedelta*10:
                    try:
                        self.__line_dict[id].set_visible(False)
                        self.__quiver_dict[id].set_visible(False)
                        self.__label_dict[id].set_visible(False)
                        self.__traj_dict[id].set_visible(False)
                    except KeyError:
                        pass
                continue
            
            # Task yet to begin: skip
            if t.start_time > new_t:
                continue
            
            nt = t.straight_update(new_t - t.start_time)
            
            # If already added via paths, skip
            if id in added_acs:
                continue
            
            # Add for drawing and logging
            output.append((nt.stats,nt.start.to_pose3D(self.transformer,True)))
            self.tasklist[i] = nt
            
            # If there is an influence circle, check if the aircraft is inside it
            if self.influence_circle is not None:
                d = distance(nt.start.latitude,nt.start.longitude,self.influence_circle.lat,self.influence_circle.lon)
                # Outside the influence circle: create a straight path (obstacle for other aircraft)
                if d > self.influence_circle.radius*NM_TO_METERS:
                    start_pose = nt.start.to_pose3D(self.transformer,True)
                    end_pose = nt.start.project(self.cmd_shift*NM_TO_METERS).to_pose3D(self.transformer,True)
                    path = Path.straight_path(start_pose,end_pose,nt.stats.airspeed)
                    solo_plan = FleetPlan(self.separation,self.z_alpha,self.wind_x,self.wind_y,path.total_length/nt.stats.airspeed,[(nt.stats,path)])
                    if self.scheduled is None:
                        self.scheduled = solo_plan
                    else:
                        self.scheduled.merge(solo_plan)
                    
                else:
                    candidate_acs.add(id)
            else:
                candidate_acs.add(id)
            
            added_acs.add(id)
        
                
        self.__t = new_t
        
        ## Update drawings
        self.__update_axes(output, new_t)
        
        # Compute which aircraft can be rescheduled
        if reschedule:
            ## Remove those with no reschedule possible
            no_reschedule:set[int] = set()
            for id in candidate_acs:
                if self.schedule_counters[id] >= self.max_reschedule:
                    no_reschedule.add(id)
                    continue
                try:
                    last_schedule = self.__last_schedules[id]
                    if new_t - last_schedule <= reschedule_threshold:
                        no_reschedule.add(id)
                        continue
                except KeyError:
                    pass
            
                if self.tasklist[self.__task_index[id]].end_time - new_t <= final_time:
                    no_reschedule.add(id)
                    continue
                    
                
            candidate_acs = candidate_acs-no_reschedule
            
            if len(candidate_acs) == 0:
                reschedule = False
        
        # If some aircraft have to be rescheduled, do it
        if reschedule:
            ## Print to JSON the set paths
            obstacles_exist = False
            if self.scheduled is not None:
                print(f"Plan time: {self.scheduled.duration}")
                noncandidate_acs = set(self.scheduled._traj_dict.keys()) - candidate_acs
                
                
                if len(noncandidate_acs) > 0:
                    print_FleetPlan_to_JSON(self.obstacles_json_path,self.scheduled,noncandidate_acs,True)
                
                    obstacles_exist = True
            
            
            ## Print to CSV the problems to solve
            pp_problems = []
            for id in candidate_acs:
                i = self.__task_index[id]
                task = self.tasklist[i]
                ppp = task.to_AC_PP_Problem(self.transformer,self.timeshifts)
                if self.scheduled is not None and id in self.scheduled._traj_dict.keys():
                    _,p = self.scheduled.get_path(id)
                    ppp.start = p.start
                pp_problems.append(ppp)
                self.schedule_counters[id] += 1
                
            write_pathplanning_problem_to_CSV(self.input_csv_path,pp_problems,True)
            
            ## Call the solver
            try:
                self.__last_global_schedule = new_t
                solve_problem(
                    self.solver_path,
                    self.input_csv_path,
                    self.output_json_path,
                    self.separation,
                    (self.wind_x,self.wind_y),
                    threads,
                    self.obstacles_json_path if obstacles_exist else None,
                    self.geometric_obstacle_json_path if self.geometric_obstacle_json_path is not None and self.geometric_obstacle_json_path.exists() else None,
                    [self.cmd_shift],
                    [self.threshold_shift]
                )
                
                ## Parse the result and merge
            
                solved = parse_trajectories_from_JSON(self.output_json_path)
                
                if self.scheduled is None:
                    self.scheduled = solved
                else:
                    self.scheduled.merge(solved)
                    
                ## Update the task ends with the planning results
                for s,p in self.scheduled.trajectories:
                    i = self.__task_index[s.id]
                    dduration = (pd.Timedelta(minutes=p.duration()) - (self.tasklist[i].end_time - new_t)).total_seconds()/60
                    self.tasklist[i].end_time = pd.Timedelta(minutes=p.duration()) + new_t
                    dinit_duration = (self.tasklist[i].end_time - self.tasklist[i].initial_end_time).total_seconds()/60
                    if abs(dduration) > 0.1:
                        print(f"Change in arrival for {s.id}: {abs(dinit_duration):.1f} min {'earlier' if dduration < 0 else 'later'}")
                    
                    if self.__axes is not None:
                        poses = [p.pose_at(t) for t in np.linspace(0,p.duration(),50,endpoint=True)]
                        
                            
                        c = self.__color_dict[s.id]
                        _,l,_,_ = plot_pose2d_sequence(self.__axes,poses,False,False,alpha=0.2,color=c,linestyle=':')
                        try:
                            self.__traj_dict[s.id].set_visible(False)
                            del self.__traj_dict[s.id]
                        except KeyError:
                            pass
                        self.__traj_dict[s.id] = l[0]
                        
                    
            except Exception as e:
                for id in candidate_acs:
                    self.schedule_counters[id] -= 1
                self.__encountered_exception = e
                print(f"EXCEPTION: {e}")
                
                
        if self.__axes is not None:
            self.__axes.set_title(str(new_t))
            plt.pause(0.1)
            
        return output

    def __update_axes(self, output:list[tuple[ACStats,Pose3D]], new_t:pd.Timestamp):
        if self.__axes is None:
            return
        
        for s,p in output:
            id = s.id
            try:
                line = self.__line_dict[id]
                xs = np.append(line.get_xdata(),p.x)
                ys = np.append(line.get_ydata(),p.y)
                line.set_xdata(xs[-self.line_buffersize:])
                line.set_ydata(ys[-self.line_buffersize:])
            except KeyError:
                xs = [p.x]
                ys = [p.y]
                color = self.__color_dict[id]
                line = self.__axes.plot(xs,ys,alpha=0.2,marker=',',color=color)[0]
            self.__line_dict[id] = line
            color = line.get_color()
                
            try:
                endpoint = self.__pos_dict[id]
            except KeyError:
                endpoint = self.__axes.plot([p.x],[p.y],marker='^',markerfacecolor=(0,0,0,0),markeredgecolor=color)[0]
                self.__pos_dict[id] = endpoint
                
            endpoint.set_xdata([p.x])
            endpoint.set_ydata([p.y])
                
                
            try:
                quiver = self.__quiver_dict[id]
            except KeyError:
                quiver = self.__axes.quiver([p.x],[p.y],[np.cos(p.theta)*s.airspeed],[np.sin(p.theta)*s.airspeed],angles='xy',pivot='tail',color=color,
                                                headwidth=0,headlength=0,headaxislength=0,width=0.002)
            quiver.set_offsets([p.x,p.y])
            quiver.set_UVC([np.cos(p.theta)],[np.sin(p.theta)])
            self.__quiver_dict[id] = quiver
                
                
            try:
                text = self.__label_dict[id]
            except KeyError:
                text = self.__axes.text(p.x,p.y,str(id))
                self.__label_dict[id] = text
            text.set_position((p.x,p.y))
            t = self.tasklist[self.__task_index[id]]
            text.set_text(f"  {id}: T -{(t.end_time-new_t).total_seconds()/60:.1f} min")
                
            
        if self.__min_dist_line is not None and len(output) >= 2:
            min_dist, index1, index2 = min_XY_dist(list(p for _,p in output))
            stat1,p1 = output[index1]
            stat2,p2 = output[index2]
                
            label=f"Min distance ({stat1.id},{stat2.id}) : {min_dist:.2f} NM"
            if min_dist < self.separation:
                label += "\n!!! LOSS OF SEPARATION !!!"
                
            self.__min_dist_line.set_xdata([p1.x,p2.x])
            self.__min_dist_line.set_ydata([p1.y,p2.y])
            self.__min_dist_line.set_label(label)
                
            self.__axes.legend(*self.__axes.get_legend_handles_labels())
    
    def simulate(self, timestep:pd.Timedelta,
                 time_threshold:pd.Timedelta, reschedule_threshold:pd.Timedelta,
                 ac_num_threshold:int,
                 final_time:pd.Timedelta,
                 threads:int):
        # log = []
        
        while self.__t < self.__end_of_times:
            try:
                do_schedule = False
                if (self.__t - self.__last_global_schedule) >= time_threshold:
                    do_schedule = True
                
                if not(do_schedule):
                    unscheduled = 0
                    for task in self.tasklist:
                        if task.start_time <= self.__t and task.end_time > self.__t:
                            if self.scheduled is None:
                                unscheduled += 1
                            else:
                                if task.stats.id not in self.scheduled._traj_dict.keys():
                                    unscheduled += 1
                    
                    do_schedule = unscheduled >= ac_num_threshold
                
                
                poss = self.step(timestep,reschedule_threshold,final_time,do_schedule,threads)
                # log.append((self.__t,poss))
                # if self.__encountered_exception is not None:
                    # break
            except KeyboardInterrupt:
                print("Simulation interrupted by user.")
                break
            
        plt.ioff()
        self.report()
        
    def report(self):
        dts = []
        for task in self.tasklist:
            # Consider only tasks ended
            if task.end_time < self.__t:
                dts.append((task.end_time - task.initial_end_time).total_seconds()/60)
        
        hist,edges = np.histogram(dts)
        mean = np.mean(dts)
        median = np.median(dts)
        std = np.std(dts)
        
        fig,ax = plt.subplots(1,1)
        ax.bar(edges[:-1], hist, width=np.diff(edges), align='edge')
        ax.set_xlabel('Delay (minutes)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Flight Delays ({} flights)'.format(len(dts)))
        ax.vlines(mean, ymin=0, ymax=max(hist), colors='k', linestyles='dashed', label='Mean: {:.2f} min'.format(mean))
        ax.vlines(median, ymin=0, ymax=max(hist), colors='k', linestyles='solid', label='Median: {:.2f} min'.format(median))
        ax.vlines([mean + std, mean - std], ymin=0, ymax=max(hist), colors='g', linestyles='dotted', label='Mean $\\pm$ 1 Std Dev: {:.2f} min'.format(std))
        ax.legend()
        
        fig.show()
        


#################### Entrypoint ####################


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        'Landing scheduler'
    )
    parser.add_argument('solver',help='Location of the DubinsFleetPlanner solver')
    
    parser.add_argument('-d','--mdist',type=float,help='Minimal distance separating aircraft, in NM. Default to 4.',default=4)
    parser.add_argument('-v','--speed',type=float,help='Nominal speed for aircraft, in knots. Default to 250.',default=250)
    parser.add_argument('-r','--turn-radius',dest='turn_radius',
        type=float, help='Minimal turn radius for all aircraft. Default to 1.33 NM (full turn in 2 minutes at 250 kt)',default=1.33)
    parser.add_argument('-w','--wind',dest='wind', nargs=2,
                        help='XY Wind, as a pair of values (X,Y). Default to (0,0), i.e. no wind.',
                        default=(0,0))
    
    parser.add_argument('--data',help='A Traffic-compatible data file. If not set, use the sample dataset of Traffic',default=None)
    parser.add_argument('--ICAOs',nargs='+',
                        help="List of ICAO codes for airports in which to look for landings in the given dataset. Default to LFPO,LFPG,LFPB",
                        default=["LFPO","LFPG","LFPB"])
    parser.add_argument('-s','--threshold-shift',type=float,dest='threshold_shift',
                        help='Distance relative to runway threshold for defining last straight, in NM. Default to 10.',default=10)
    parser.add_argument('--cmd-shift',type=float,dest='cmd_shift',
                        help="Duration (in minutes) for which no commands should issued (that is, go straight). Default to 2 minutes.", default=2)
    parser.add_argument('--influence-radius',type=float,dest='influence_radius',
                        help="Radius of influence from the airports barycenter, in NM. If set, aircraft will be scheduled only within this radius. Disabled by default.",default=None)
    parser.add_argument('--epsg',type=int,help="EPSG code for XY projection of (lat,lon) coordinates from WGS84. Default to the UTM zone of the first airport in the ICAO list.",
                        default=None)
    
    parser.add_argument('-ts','--timestep',type=float,
                        help="Simulation time step, in minutes. Default to 1/6 (ie 10s).",default=1/6)
    parser.add_argument('-I','--intervals',nargs=3,
                        help="Triplet (min,max,step) defining the offsets to the reference time for the target times. Defined in minutes. Default to (-10,10,1)",
                        default=(-10,10,1))
    parser.add_argument('-m','--max-reschedule',dest="max_reschedule",type=int,
                        help="Maximum number of times a flight can be rescheduled. Default to 3.",default=3)
    parser.add_argument('-n','--ac-threshold',dest='ac_threshold',type=int,
                        help="Minimal number of unscheduled aircraft triggering a rescheduling. Default to 1.", default=1)
    parser.add_argument('-t','--time-threshold',dest='time_threshold',type=float,
                        help="Duration ellapsed after which we reschedule aircraft. Default to 10 minutes.",default=10)
    parser.add_argument('-f','--final-time',dest='final_time',type=float,
                        help="Final duration in minutes. If an aircraft is less than this duration away from landing, it cannot be rescheduled. Default to 5 minutes.",
                        default=5)
    parser.add_argument('--threads',dest="threads",type=int,
                        help="Number of threads to be used by the solver. 0 allows it to autoselect. Default to 0.",default=0)
    
    args = parser.parse_args()

    solver = args.solver
    print("===== Parsing traffic... =====\n")
    if args.data is None:
        traffic_iter = [samples.quickstart]
    else:
        traffic_iter = iter_flightdata_by_day(args.data,id_column='flight_id')
        if traffic_iter is None:
            print("ERROR: Importing traffic failed. Exiting")
            exit(1)
    
    airports = get_airports(args.ICAOs)
    if args.epsg is None:
        transformer = get_airport_latlon_transformer(airports[0])
    else:
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{args.epsg}")

    expected_speed = args.speed # kts
    threshold_shift = args.threshold_shift # NM
    timeshifts = pd.timedelta_range(f"{args.intervals[0]} minute",f"{args.intervals[1]} minute",freq=f"{float(args.intervals[2])*60}s").to_list()

    ## Generate borders ##
    geometric_obstacle_json_path = None
    influence_circle = None
    if args.influence_radius is not None:
        geometric_obstacle_json_path = pathlib.Path("geometric_obstacles.json")
        avgx,avgy = 0,0
        for a in airports:
            x,y = transformer.transform(a.latitude,a.longitude)
            avgx += x
            avgy += y
        avgx /= len(airports)
        avgy /= len(airports)
        make_circle_boundary_file(avgx/NM_TO_METERS,avgy/NM_TO_METERS,args.influence_radius,geometric_obstacle_json_path)
        influence_circle = InfluenceCircle.from_xy_meters(avgx,avgy,args.influence_radius*NM_TO_METERS,transformer)

    for traffic in traffic_iter:
        print(f"Processing traffic for day {traffic.data['timestamp'].min().date()}")
        filtered_traffic = filter_traffic(traffic,airports)
        if filtered_traffic is None:
            print("No flights found for the given ICAO codes. Skipping to next day.")
            continue
        plt.ion()
        fig,ax = plt.subplots(figsize=(16/1.5,9/1.5))
        ax.set_aspect('equal')
        fig.tight_layout()
        
        endpoints:list[FlightEndpoints] = []
        
        for i,flight in enumerate(filtered_traffic):
            stats = ACStats(
                i,
                expected_speed/60, # Convert from kts (NM/h) to NM/minute
                1.,
                expected_speed/(60*np.pi) # Full circle in 2 minutes
            )
            
            r = extract_flight_endpoints(flight,airports,stats,0.)
            if r is None:
                continue
            else:
                if r[1] is not None:
                    endpoints.append(r[1])
                    
        print("\n===== Traffic parsing done! =====\nSetting up simulator...")
        
        sim = ArrivalsSimulator(pathlib.Path(solver),
                                transformer,
                                timeshifts,
                                threshold_shift,
                                args.cmd_shift*expected_speed/60,
                                endpoints,
                                max_reschedule=args.max_reschedule,
                                separation=args.mdist,
                                wind_x=float(args.wind[0]),
                                wind_y=float(args.wind[1]))
        
        sim.attach_axes(ax)
        
        if geometric_obstacle_json_path is not None:
            sim.geometric_obstacle_json_path = geometric_obstacle_json_path
        if influence_circle is not None:
            sim.influence_circle = influence_circle
            x,y,_ = influence_circle.to_xy_NM(transformer)
            # The plot uses NM, it is easier for distance comparison.
            ax.add_patch(
                Circle((x,y),radius=influence_circle.radius,fill=False,linestyle='--',color='k',label=f"Influence radius ({influence_circle.radius:.1f} NM)")
            )
        
        
        # try:
        #     b = float(args.border)
            
        #     xmin,xmax = ax.get_xlim()
        #     ymin,ymax = ax.get_ylim()
            
        #     make_box_boundary_file(xmin+b,xmax-b,ymin+b,ymax-b,sim.geometric_obstacle_json_path)
        #     ax.plot(
        #         [xmin+b,xmin+b,xmax-b,xmax-b,xmin+b],
        #         [ymin+b,ymax-b,ymax-b,ymin+b,ymin+b],
        #         'k-',label='Border'
        #     )
        # except ValueError:
        #     sim.geometric_obstacle_json_path = args.border
        
        ## Run simulation ##
        
        
        
        sim.simulate(pd.Timedelta(f"{args.timestep} minute"),
            pd.Timedelta(f"{args.time_threshold} minute"),
            pd.Timedelta(f"{args.time_threshold/3} minute"),
            args.ac_threshold,
            pd.Timedelta(f"{args.final_time} minute"),
            args.threads
        )
        
        cont = input("Enter 'y' to restart simulation with the next day data, quit otherwise: ")
        if cont != 'y':
            break
    
if __name__ == '__main__':
    main()
