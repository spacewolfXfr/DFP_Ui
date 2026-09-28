#!/usr/bin/python3

from __future__ import annotations

import dataclasses
from time import time
import typing
from typing import Optional
import subprocess
import pathlib
import json
import copy
from concurrent.futures import ProcessPoolExecutor
import itertools

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
from traffic.data.basic.airports import Airports

from DubinsFleetPlanner.ioUtils import AC_PP_Problem,ACStats,write_pathplanning_problem_to_CSV,print_FleetPlan_to_JSON,parse_trajectories_from_JSON
from DubinsFleetPlanner.Dubins import Path,BasicPath,FleetPlan,DubinsMove
from DubinsFleetPlanner.Poses import Pose2D,Pose3D,min_XY_dist,poses_dist,poses_dist_2D

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
    cmd.append('0.6')
    # cmd.append('0.7')
    # cmd.append('0.8')
    
    cmd.append('--ellipse-border-only')
    
    cmd.append('-v')
    cmd.append('2')
    
    cmd.append('--straights-only')
    cmd.append('--allow-shortest')
    # cmd.append('--fixed-radius')
    
    if obstacle_path is not None:
        cmd.append('-O')
        cmd.append(str(obstacle_path.resolve()))
        
    if geometric_obstacles_path is not None:
        cmd.append('-G')
        cmd.append(str(geometric_obstacles_path.resolve()))
        
    if len(start_extensions) > 0:
        cmd.append('--extend-start')
        for e in start_extensions:
            cmd.append(str(e))
    
    if len(end_extensions) > 0:
        cmd.append('--extend-end')
        for e in end_extensions:
            cmd.append(str(e))
    
    print(cmd)
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
    reschedule_threshold:pd.Timedelta   # How much time must separe two reschedulings of the same aircraft, in pd.Timedelta
    final_time:pd.Timedelta             # How much time must be left before the end of the task to allow rescheduling, in pd.Timedelta
    
    tasklist:list[FlightEndpoints] = dataclasses.field(default_factory=list)  # List of tasks to be done, sorted by start_time
    removed_ids:set[int] = dataclasses.field(default_factory=set) # Storage for aircraft (id) removed due to minimal separation violation.

    ## Temporary solver files
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
    nopath_counter:dict[int,int] = dataclasses.field(init=False) # Dict from AC id to number of times no path was found
    line_buffersize:int = 20 # Number of past points kept in visualisation
    __schedules_count:int = 0 # Number of calls to the solver
    
    ## Timekeeping
    __last_global_schedule:pd.Timestamp     = dataclasses.field(init=False)
    __last_schedules:dict[int,pd.Timestamp] = dataclasses.field(init=False)
    __task_index:dict[int,int]    = dataclasses.field(init=False) # Reverse accessor from Aircraft ID to tasklist index
    __t:pd.Timestamp              = dataclasses.field(init=False)
    __end_of_times:pd.Timestamp   = dataclasses.field(init=False)
    __encountered_exception:typing.Optional[Exception] = dataclasses.field(default=None,init=False)
    
    ## Plotting
    __axes:typing.Optional[Axes]    = dataclasses.field(default=None,init=False)
    __line_dict:dict[int,Line2D]    = dataclasses.field(default_factory=dict,init=False)
    __pos_dict:dict[int,Line2D]     = dataclasses.field(default_factory=dict,init=False)
    __traj_dict:dict[int,Line2D]    = dataclasses.field(default_factory=dict,init=False)
    __quiver_dict:dict[int,Quiver]  = dataclasses.field(default_factory=dict,init=False)
    __dest_dict:dict[tuple[str,str],Line2D]    = dataclasses.field(default_factory=dict,init=False)
    __color_dict:dict[int,Colortype]= dataclasses.field(default_factory=dict,init=False)
    __label_dict:dict[int,Text]     = dataclasses.field(default_factory=dict,init=False)
    __min_dist_line:typing.Optional[Line2D] = dataclasses.field(default=None,init=False)
    
    @property
    def schedules_count(self) -> int:
        return self.__schedules_count
    
    def __setup_sim(self):
        assert self.tasklist is not None and len(self.tasklist) > 0, "No tasks to simulate"
        
        self.schedule_counters = dict()
        self.nopath_counter = dict()
        
        self.__t = self.tasklist[0].start_time
        self.__last_global_schedule = self.__t
        self.__end_of_times = max(t.end_time for t in self.tasklist)
        self.__task_index = dict()
        self.__encountered_exception = None
        self.__last_schedules = dict()
        self.__schedules_count = 0
        
        self.__axes = None
        
        for i,t in enumerate(self.tasklist):
            self.__task_index[t.stats.id] = i
            self.schedule_counters[t.stats.id] = 0
        
        if self.scheduled is not None:
            self.separation = self.scheduled.separation
            self.z_alpha = self.scheduled.z_alpha
            self.wind_x = self.scheduled.wind_x
            self.wind_y = self.scheduled.wind_y
    
    
    def get_task(self,ac_id:int) -> FlightEndpoints:
        return self.tasklist[self.__task_index[ac_id]]
    
    def is_running(self,ac_id:int) -> bool:
        task = self.get_task(ac_id)
        return task.start_time <= self.__t and task.end_time > self.__t
    
    def is_scheduled(self,ac_id:int) -> bool:
        t = self.get_task(ac_id)
        return t.planned
    
    def can_be_scheduled(self,task:FlightEndpoints,path:Path) -> bool:        
        # If it has a plan, don't schedule by default
        if task.planned:
            add_me = False
            
            # Check if geometrically possible
            can_be_rescheduled = False
            # Reschedule only possible during a straight without incoming turn
            if path.sections[0].type == DubinsMove.STRAIGHT:
                if len(path.junctions) > 0:
                    if path.junctions[0] > self.reschedule_threshold.total_seconds()/60:
                        can_be_rescheduled = True
                else:
                    if path.duration() > self.reschedule_threshold.total_seconds()/60:
                        can_be_rescheduled = True
            
            # If geometrically feasible, consider if it is close to end
            if can_be_rescheduled:
                add_me = True
                # If too close, don't reschedule
                if task.end_time - self.__t <= self.final_time:
                    add_me = False
                    
                # Limit rescheduling number/frequency    
                id = task.stats.id
                if self.schedule_counters[id] >= self.max_reschedule:
                   add_me = False
                try:
                   last_schedule = self.__last_schedules[id]
                   if self.__t - last_schedule <= self.reschedule_threshold:
                       add_me = False
                except KeyError:
                   pass                        
        else:
            # Schedule if it does not have a plan yet
            add_me = True
            # Except if outside the influence circle
            if self.__xy_circle is not None:
                dx = path.start.x - self.__xy_circle[0]
                dy = path.start.y - self.__xy_circle[1]
                if dx*dx + dy*dy > self.__xy_circle[2]*self.__xy_circle[2]:
                    add_me = False
        
        return add_me
                

    def step(self,timedelta:pd.Timedelta,
             threads:int=0) -> tuple[list[tuple[ACStats,Pose3D]],bool]:
        output:list[tuple[ACStats,Pose3D]] = []
        
        ### Time forward
        new_t = timedelta + self.__t
        print(new_t.strftime("%Y-%m-%d %H:%M:%S"))
        
        ### Metadata
        if self.influence_circle is not None and self.__xy_circle is None:
            self.__xy_circle = self.influence_circle.to_xy_NM(self.transformer)
        
        
        ## Check task to be done
        unplanned_task:set[int] = set()
        new_paths:dict[int,tuple[ACStats,Path]] = {}
        for i,t in enumerate(self.tasklist):
            id = t.stats.id
            
            # Task is removed due to loss of separation: skip it
            if id in self.removed_ids:
                continue
            
            # Task already ended: remove drawing
            if t.end_time <= new_t:
                if (new_t - t.end_time) > timedelta*10:
                    self.__remove_from_axes(id)
                continue
            
            # Task yet to begin: break (since tasklist is sorted by start_time)
            if t.start_time > new_t:
                break
            
            # If the task is planned, skip it
            if t.planned: 
                continue
            else:
                unplanned_task.add(id)
            
            # Otherwise, create a straight plan for it
            start_pose = None
            if self.scheduled is not None:
                try:
                    _,p = self.scheduled.get_path(id)
                    start_pose = p.start
                except KeyError:
                    pass
                
            if start_pose is None:
                start_pose = t.start.to_pose3D(self.transformer,True)
            start_pose.z = 0
            end_pose = copy.copy(start_pose)
            end_pose.x += np.cos(start_pose.theta)*t.stats.airspeed*self.cmd_shift*2
            end_pose.y += np.sin(start_pose.theta)*t.stats.airspeed*self.cmd_shift*2
            path = Path.straight_path(start_pose,end_pose,t.stats.airspeed)
            
            new_paths[id] = (t.stats,path)
        
        # Add the temporary straight paths to the current plan, if any
        if len(new_paths) > 0:
            new_plan = FleetPlan(self.separation,self.z_alpha,self.wind_x,self.wind_y,self.cmd_shift,new_paths)
            if self.scheduled is None:
                self.scheduled = new_plan
            else:
                self.scheduled.merge(new_plan)

        ## Move forward
        if self.scheduled is not None:
            self.scheduled = self.scheduled.follow_for(timedelta.total_seconds()/60)
            for s,p in self.scheduled.trajectories.values():
                output.append((s,p.start))
                
            # Delete almost ended plan
            if self.scheduled.duration < 1e-3:
                self.scheduled = None
                
                    
        self.__t = new_t
        

        loss_of_separation = False
        if len(output) > 1:
            min_dist, index1, index2 = min_XY_dist(list(p for _,p in output))
            ac1 = output[index1][0].id
            ac2 = output[index2][0].id
            
            if min_dist < self.separation:
                loss_of_separation = True
                print(f"LOSS OF SEPARATION: {min_dist:.2f} NM between {ac1} and {ac2}. ",end='')
                t1 = self.get_task(ac1)
                t2 = self.get_task(ac2)
                if t1.planned and t2.planned:
                    print(f"Both aircraft are planned!!!")
                else:
                    if t1.planned:
                        self.removed_ids.add(ac2)
                        if self.scheduled is not None:
                            del self.scheduled.trajectories[ac2]
                            self.__remove_from_axes(ac2)
                        print(f"Removing {ac2} from the simulation")
                    else:
                        self.removed_ids.add(ac1)
                        if self.scheduled is not None:
                            del self.scheduled.trajectories[ac1]
                            self.__remove_from_axes(ac1)
                        print(f"Removing {ac1} from the simulation")
                
                
        
        ## Scheduling rounds
        
        # Add for rescheduling those who can
        candidate_acs:set[int] = set()
        if self.scheduled is not None:
            for s,p in self.scheduled.trajectories.values():
                t = self.get_task(s.id)
                
                if self.can_be_scheduled(t,p):
                    candidate_acs.add(s.id)
        
        # Compute which aircraft can be rescheduled
        if self.__t - self.__last_global_schedule < self.reschedule_threshold and len(unplanned_task) == 0:
            candidate_acs = set()
            print("Too early to reschedule")
        else:
            print(f"Candidate aircraft: {candidate_acs}")
        
        # If some aircraft have to be rescheduled, do it
        success_schedule = True
        noncandidate_acs = set()
        if len(candidate_acs) > 0:
            ## Print to JSON the set paths
            obstacles_exist = False
            if self.scheduled is not None:
                print(f"Plan time: {self.scheduled.duration}")
                noncandidate_acs = set(self.scheduled.ids) - candidate_acs
                
                if len(noncandidate_acs) > 0:
                    print_FleetPlan_to_JSON(self.obstacles_json_path,self.scheduled,noncandidate_acs,True)
                
                    obstacles_exist = True
            
            
            ## Print to CSV the problems to solve
            pp_problems = []
            for id in candidate_acs:
                i = self.__task_index[id]
                task = self.tasklist[i]
                ppp = task.to_AC_PP_Problem(self.transformer,self.timeshifts,True,False) # This forces the Z values to all be 0. Be warned when trying to implement 3D control!
                if self.scheduled is not None and id in self.scheduled.ids:
                    _,p = self.scheduled.get_path(id)
                    ppp.start = p.start
                pp_problems.append(ppp)
                
            write_pathplanning_problem_to_CSV(self.input_csv_path,pp_problems,True)
            
            ## Call the solver
            try:
                self.__schedules_count += 1
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
                self.__last_global_schedule = self.__t
                newly_scheduled = False
                for s,p in solved.trajectories.values():
                    if s.id in noncandidate_acs:
                        continue
                    t = self.get_task(s.id)
                    self.__last_schedules[s.id] = self.__t
                    self.schedule_counters[s.id] += 1
                    if not t.planned:
                        newly_scheduled = True
                    t.planned = True
                    
                
                if self.scheduled is None:
                    self.scheduled = solved
                else:
                    self.scheduled.merge(solved)
                    # if set(self.scheduled.list_ids()) != set(solved.list_ids()) or newly_scheduled:
                    #    Add new solutions to the existing plan
                    # else:
                        # if self.scheduled.sum_of_durations() > solved.sum_of_durations():
                            # self.scheduled = solved
                    
                ## Update the task ends with the planning results
                for s,p in self.scheduled.trajectories.values():
                    t = self.get_task(s.id)
                    if t.planned:
                        dduration = (pd.Timedelta(minutes=p.duration()) - (t.end_time - new_t)).total_seconds()/60
                        t.end_time = pd.Timedelta(minutes=p.duration()) + new_t
                        dinit_duration = (t.end_time - t.initial_end_time).total_seconds()/60
                        if abs(dduration) > 0.1:
                            print(f"Change in arrival for {s.id}: {abs(dinit_duration):.1f} min {'earlier' if dduration < 0 else 'later'}")
                    
                    # assert poses_dist_2D(self.tasklist[i].end.to_pose3D(self.transformer,True),p.end) < 1e-3, f"Endpoints do not match for {s.id}: {self.tasklist[i].end.to_pose3D(self.transformer,True)} vs {p.end}"
                    # p_lat,p_lon = self.transformer.transform(p.end.x*NM_TO_METERS,p.end.y*NM_TO_METERS,direction='INVERSE')
                    # assert abs(p_lat-self.tasklist[i].end.latitude) < 1e-3, f"Latitudes do not match for {s.id}: {p_lat} vs {self.tasklist[i].end.latitude}"
                    # assert abs(p_lon-self.tasklist[i].end.longitude) < 1e-3, f"Longitudes do not match for {s.id}: {p_lon} vs {self.tasklist[i].end.longitude}"
                    
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
                if isinstance(e,AssertionError):
                    raise e
                
                self.__encountered_exception = e
                success_schedule = False
                print(f"EXCEPTION: {e}")
                
                
        if self.__axes is not None:
            self.__update_axes(output, new_t, self.scheduled)
            self.__axes.set_title(str(new_t))
            
        return output, success_schedule

    def setup_simulation(self,endpoints:list[FlightEndpoints]):
        self.tasklist = endpoints
        self.tasklist.sort(key=lambda fpts : fpts.start_time)
        self.__setup_sim()

    def simulate(self,
                 timestep:pd.Timedelta,
                 threads:int):
        
        assert self.tasklist is not None and len(self.tasklist) > 0, "No tasks to simulate. Did not call `setup_simulation` first?"
        
        # log = []
        
        while self.__t < self.__end_of_times:
            try:
                states,success_schedule = self.step(timestep,threads)
                # log.append((self.__t,states))
                # if self.__encountered_exception is not None:
                    # break
            except KeyboardInterrupt:
                print("Simulation interrupted by user.")
                break
            
        plt.ioff()
        self.report()
        
    #################### Plotting ####################
        
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
        
    def __remove_from_axes(self,ac_id:int):    
        try:
            self.__pos_dict[ac_id].set_visible(False)
            del self.__pos_dict[ac_id]
        except KeyError:
            pass
        
        try:
            self.__line_dict[ac_id].set_visible(False)
            del self.__line_dict[ac_id]
        except KeyError:
            pass
        
        try:
            self.__quiver_dict[ac_id].set_visible(False)
            del self.__quiver_dict[ac_id]
        except KeyError:
            pass
        
        try:
            self.__label_dict[ac_id].set_visible(False)
            del self.__label_dict[ac_id]
        except KeyError:
            pass
        
        try:
            self.__traj_dict[ac_id].set_visible(False)
            del self.__traj_dict[ac_id]
        except KeyError:
            pass
        
    def __update_axes(self, states:list[tuple[ACStats,Pose3D]], new_t:pd.Timestamp, trajs:Optional[FleetPlan]=None) -> Optional[tuple[float,int,int]]:
        """
        Update the axes given the input states and time.

        Args:
            states (list[tuple[ACStats,Pose3D]]): _description_
            new_t (pd.Timestamp): _description_

        Returns:
            Optional[tuple[float,int,int]]: If there are at least two states, returns the minimum distance and the IDs of the two aircraft that are closest. Otherwise, returns None.
        """
        if self.__axes is None:
            return
        
        for s,p in states:
            id = s.id
            
            ## Clean trajs 
            try:
                self.__traj_dict[s.id].set_visible(False)
                del self.__traj_dict[s.id]
            except KeyError:
                pass
            
            if trajs is not None:
                try:
                    _,path = trajs.get_path(s.id)
                    poses = [path.pose_at(t) for t in np.linspace(0,path.duration(),50,endpoint=True)]
                    c = self.__color_dict[s.id]
                    _,l,_,_ = plot_pose2d_sequence(self.__axes,poses,False,False,alpha=0.2,color=c,linestyle=':')
                    self.__traj_dict[s.id] = l[0]
                except KeyError:
                    pass
            
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
            
            if not self.is_scheduled(id):
                endpoint.set_marker('P')
            else:
                endpoint.set_marker('^')
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
                
        
        
        
        if self.__min_dist_line is not None and len(states) >= 2:
            min_dist, index1, index2 = min_XY_dist(list(p for _,p in states))
            stat1,p1 = states[index1]
            stat2,p2 = states[index2]
                
            label=f"Min distance ({stat1.id},{stat2.id}) : {min_dist:.2f} NM"
            if min_dist < self.separation:
                label += "\n!!! LOSS OF SEPARATION !!!"
                
            self.__min_dist_line.set_xdata([p1.x,p2.x])
            self.__min_dist_line.set_ydata([p1.y,p2.y])
            self.__min_dist_line.set_label(label)
                
            self.__axes.legend(*self.__axes.get_legend_handles_labels())
            
            return min_dist,stat1.id,stat2.id
        else:
            return None
    
        
    def report(self):
        dts = []
        for task in self.tasklist:
            # Consider only tasks ended and not removed due to loss of separation
            if task.end_time < self.__t and task.stats.id not in self.removed_ids:
                dts.append((task.end_time - task.initial_end_time).total_seconds()/60)
        
        min_dt = np.floor(np.min(dts))
        max_dt = np.ceil(np.max(dts))
        edges = [min_dt-1] + list(range(-15,15,2)) + [max_dt+1]
        
        hist,edges = np.histogram(dts,edges)
        mean = np.mean(dts)
        median = np.median(dts)
        std = np.std(dts)
        
        fig,ax = plt.subplots(1,1)
        bars = ax.bar(edges[:-1], hist, width=np.diff(edges), align='edge')
        ax.bar_label(bars, fontsize=20, color='navy')
        ax.set_xlabel('Delay (minutes)')
        ax.set_xticks(edges)
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Flight Delays ({} flights ; {} removed)'.format(len(dts), len(self.removed_ids)))
        ax.vlines(mean, ymin=0, ymax=max(hist), colors='k', linestyles='dashed', label='Mean: {:.2f} min'.format(mean))
        ax.vlines(median, ymin=0, ymax=max(hist), colors='k', linestyles='solid', label='Median: {:.2f} min'.format(median))
        ax.vlines([mean + std, mean - std], ymin=0, ymax=max(hist), colors='g', linestyles='dotted', label='Mean $\\pm$ 1 Std Dev: {:.2f} min'.format(std))
        ax.legend()
        
        fig.show()
        


#################### Entrypoint ####################


def main():
    import argparse,cProfile
    
    parser = argparse.ArgumentParser(
        'Landing scheduler'
    )
    parser.add_argument('solver',help='Location of the DubinsFleetPlanner solver')
    
    parser.add_argument('-d','--mdist',type=float,help='Minimal distance separating aircraft, in NM. Default to 3.',default=3)
    parser.add_argument('-v','--speed',type=float,help='Nominal speed for aircraft, in knots. Default to 200.',default=200)
    parser.add_argument('-r','--turn-radius',dest='turn_radius',
        type=float, help='Minimal turn radius for all aircraft. Default to 1.33 NM (full turn in 2 minutes at 250 kt)',default=1.33)
    # parser.add_argument('-w','--wind',dest='wind', nargs=2,
    #                     help='XY Wind, as a pair of values (X,Y). Default to (0,0), i.e. no wind.',
    #                     default=(0,0))
    
    parser.add_argument('--data',help='A Traffic-compatible data file. If not set, use the sample dataset of Traffic',default=None)
    parser.add_argument('--ICAOs',nargs='+',
                        help="List of ICAO codes for airports in which to look for landings in the given dataset. Default to LFPO,LFPG,LFPB",
                        default=["LFPO","LFPG","LFPB"])
    parser.add_argument('-s','--threshold-shift',type=float,dest='threshold_shift',
                        help='Distance relative to runway threshold for defining last straight, in NM. Default to 10.',default=10)
    parser.add_argument('--cmd-shift',type=float,dest='cmd_shift',
                        help="Duration (in minutes) for which no commands should issued (that is, go straight). Default to 1 minute.", default=1)
    parser.add_argument('--influence-radius',type=float,dest='influence_radius',
                        help="Radius of influence from the airports barycenter, in NM. If set, aircraft will be scheduled only within this radius. Disabled by default.",default=None)
    parser.add_argument('--epsg',type=int,help="EPSG code for XY projection of (lat,lon) coordinates from WGS84. Default to the UTM zone of the first airport in the ICAO list.",
                        default=None)
    
    parser.add_argument('-ts','--timestep',type=float,
                        help="Simulation time step, in minutes. Default to 1/6 (ie 10s).",default=1/6)
    parser.add_argument('-I','--intervals',nargs=3,
                        help="Triplet (min,max,step) defining the offsets to the reference time for the target times. Defined in minutes. Default to (-10,10,2)",
                        default=(-10,10,2))
    parser.add_argument('-m','--max-reschedule',dest="max_reschedule",type=int,
                        help="Maximum number of times a flight can be rescheduled. Default to 3.",default=3)
    parser.add_argument('-t','--time-threshold',dest='time_threshold',type=float,
                        help="Duration ellapsed after which we reschedule aircraft. Default to 5 minutes.",default=5)
    parser.add_argument('-f','--final-time',dest='final_time',type=float,
                        help="Final duration in minutes. If an aircraft is less than this duration away from landing, it cannot be rescheduled. Default to 5 minutes.",
                        default=5)
    parser.add_argument('--threads',dest="threads",type=int,
                        help="Number of threads to be used by the solver. 0 allows it to autoselect. Default to 0.",default=0)
    parser.add_argument('--profile',dest="profile",action='store_true',
                        help="If set, profile the simulation and print the results at the end. Default to False.",default=False)
    parser.add_argument('--no-ui',dest="no_ui",action='store_true',
                        help="If set, do not display the simulation UI. Default to False.",default=False)
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
        print(filtered_traffic)
        
        if not args.no_ui:
            plt.ion()
            fig,ax = plt.subplots(figsize=(16/1.5,9/1.5))
            ax.set_aspect('equal')
            fig.tight_layout()
        
        endpoints:list[FlightEndpoints] = generate_flightEndpoints(airports, expected_speed, filtered_traffic)
        # for ep in endpoints:
            # ax.scatter(ep.start.to_pose3D(transformer,True).x,ep.start.to_pose3D(transformer,True).y,marker='o',color='k',alpha=0.2)
            # ax.scatter(ep.end.to_pose3D(transformer,True).x,ep.end.to_pose3D(transformer,True).y,marker='x',color='k',alpha=0.2)
                    
        print("\n===== Traffic parsing done! =====\nSetting up simulator...")
        
        sim = ArrivalsSimulator(pathlib.Path(solver),
                                transformer,
                                timeshifts,
                                threshold_shift,
                                args.cmd_shift*expected_speed/60,
                                pd.Timedelta(f"{args.time_threshold} minute"),
                                pd.Timedelta(f"{args.final_time} minute"),
                                max_reschedule=args.max_reschedule,
                                separation=args.mdist,
                                wind_x=0.,#float(args.wind[0]),
                                wind_y=0.,#float(args.wind[1]))
        )
        
        if geometric_obstacle_json_path is not None:
            sim.geometric_obstacle_json_path = geometric_obstacle_json_path
        if influence_circle is not None:
            sim.influence_circle = influence_circle
            x,y,_ = influence_circle.to_xy_NM(transformer)
            # The plot uses NM, it is easier for distance comparison.
            if not args.no_ui:
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
        
        sim.setup_simulation(endpoints)
        
        if not args.no_ui:
            sim.attach_axes(ax)
        
        start_time = time()
        
        if args.profile:
            print("Profiling simulation...")
            cProfile.runctx('sim.simulate(pd.Timedelta(f"{args.timestep} minute"), args.threads)', globals(), locals(),filename='simulation_profile.prof')
        else:
            sim.simulate(pd.Timedelta(f"{args.timestep} minute"), args.threads)
        
        end_time = time()
        print(f"Simulation time: {end_time - start_time:.2f} seconds. Called solver {sim.schedules_count} times.")
        cont = input("Enter 'y' to restart simulation with the next day data, quit otherwise: ")
        if cont != 'y':
            break

def __wrapped_extract_flight_endpoints(args):
    flight, airports, expected_speed = args
    
    id = flight.flight_id.split('_')[1]
    
    stats = ACStats(
                int(id),
                expected_speed/60, # Convert from kts (NM/h) to NM/minute
                1.,
                expected_speed/(60*np.pi) # Full circle in 2 minutes
            )
    
    
    o = extract_flight_endpoints(flight,airports,stats,0)
    return o[1] if o is not None and o[1] is not None else None

def generate_flightEndpoints(airports:Airports, expected_speed:float, filtered_traffic:Traffic) -> list[FlightEndpoints]:
    endpoints = []
    
    number_of_flights = filtered_traffic.flight_ids.__len__()
        
    args_it = zip(filtered_traffic.iterate(),
                  itertools.repeat(airports, number_of_flights),
                  itertools.repeat(expected_speed, number_of_flights))

    
    
    with ProcessPoolExecutor() as executor:
        endpoints = list(executor.map(__wrapped_extract_flight_endpoints, args_it))
    
    endpoints = [ep for ep in endpoints if ep is not None]
    
    
    return endpoints
    
if __name__ == '__main__':
    main()
