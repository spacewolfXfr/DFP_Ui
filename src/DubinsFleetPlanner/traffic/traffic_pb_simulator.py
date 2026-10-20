#!/usr/bin/python3

from __future__ import annotations

import dataclasses
from time import time
import datetime
import typing
from typing import Optional
import subprocess
import pathlib,tempfile
import json,pickle
import copy
from enum import IntEnum
from argparse import Namespace

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.quiver import Quiver
from matplotlib.text import Text
from matplotlib.patches import Circle
from matplotlib.ticker import FuncFormatter
from matplotlib import colormaps as cm

import pandas as pd

from traffic.core import Traffic
from traffic.data import samples

from DubinsFleetPlanner.ioUtils import AC_PP_Problem,ACStats,write_pathplanning_problem_to_CSV,print_FleetPlan_to_JSON,parse_trajectories_from_JSON
from DubinsFleetPlanner.Dubins import Path,BasicPath,FleetPlan,DubinsMove
from DubinsFleetPlanner.Poses import Pose2D,Pose3D,min_XY_dist,poses_XY_dist
from DubinsFleetPlanner.SpeedProfile import SpeedProfile,min_tdist

from pyproj import Transformer

from .AirportHelpers import NM_TO_METERS, InfluenceCircle, get_airports, get_airport_latlon_transformer,get_other_runway_name,get_runway
from .FlightExtraction import FlightEndpoints,LatlonPose,generate_flightEndpoints
from .TrafficReader import iter_flight_by_period,filter_traffic
from .TrafficPlotter import make_colordict
from .TaskResultsPlotter import plot_endpoints_results,SimulationSummary

from DubinsFleetPlanner.UI.plotting import plot_pose2d_sequence,ColorType

#################### Utility ####################

class VerbosityLevel(IntEnum):
    SILENT = 0
    NORMAL = 1
    VERYVERBOSE = 2
    VERYVERYVERBOSE = 3

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
            
#################### Call the underlying solver ####################


def solve_problem(solver:pathlib.Path,src_dir:pathlib.Path,dest_dir:pathlib.Path,
                   separation:float, tdist:bool,
                   wind:tuple[float,float],threads:int,
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
    
    if tdist:
        cmd.append('--tdist')
    
    # cmd.append('--samples')
    # cmd.append('5')
    
    # cmd.append('--ellipse')
    # cmd.append('0.5')
    # cmd.append('0.7')
    # cmd.append('0.8')
    
    cmd.append('--ellipse-border-only')
    
    cmd.append('-v')
    cmd.append('2')
    
    cmd.append('--straights-only')
    cmd.append('--allow-shortest')
    cmd.append('--fixed-radius')
    
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
    
    print("\n  Sending new problem to solver")
    print(cmd)
    output = subprocess.run(cmd)
    # print(output)
    # output.check_returncode()
    return output

#################### Ongoing flights simulator ####################

@dataclasses.dataclass
class ArrivalsSimulator:
    solver_path:pathlib.Path
    transformer:Transformer  # Convert (lat,lon) coordinates from WGS84 into some (x,y) coordinates for the solver. Must be a Conformal projection (preserve angles)
    timeshifts:list[pd.Timedelta]
    threshold_shift:float   # By how much the endpoint have been moved along a straight line, in NM
    cmd_shift:float         # Minimal straight length at the start of newly planned trajectory, in NM
    reschedule_threshold:pd.Timedelta   # How much time must separe two reschedulings of the same aircraft, in pd.Timedelta
    final_time:pd.Timedelta             # How much time must be left before the end of the task to allow rescheduling, in pd.Timedelta
    lookahead_time:pd.Timedelta = pd.Timedelta(minutes=30)        # How far in the future to look for incoming or departing flights to be added as obstacles, in pd.Timedelta
    
    tasklist:list[FlightEndpoints] = dataclasses.field(default_factory=list)  # List of tasks to be done, sorted by start_time

    ## Solver files
    geometric_obstacle_json_path:Optional[pathlib.Path]   = None # Given by the user
    # The others are dynamically generated
    
    verbosity:VerbosityLevel = VerbosityLevel.NORMAL
    
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
    line_buffersize:int = 10 # Number of past points kept in visualisation
    __schedules_count:int = 0 # Number of calls to the solver
    timedistance:bool = False # If True, interpret separation as time-based separation (in minutes, to be multiplied by the speed of the aircraft). If False, interpret separation as distance-based separation (in NM).
    __traffic_log:list[dict[str,typing.Any]] = dataclasses.field(default_factory=list,init=False,repr=False)
    
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
    __color_dict:dict[int,ColorType]= dataclasses.field(default_factory=dict,init=False)
    __label_dict:dict[int,Text]     = dataclasses.field(default_factory=dict,init=False)
    __min_dist_line:typing.Optional[Line2D] = dataclasses.field(default=None,init=False)
    
    @property
    def schedules_count(self) -> int:
        return self.__schedules_count    

    def __log_traffic_states(self, states:list[tuple[ACStats,Pose3D,float]], timestamp:pd.Timestamp) -> None:
        for stats, pose, speed in states:
            task = self.get_task(stats.id)
            latlon = LatlonPose.from_pose3D(pose, self.transformer, from_NM=True)
            self.__traffic_log.append({
                "timestamp": timestamp,
                "icao24": task.icao24,
                "callsign": f"SIM{stats.id}",
                "latitude": latlon.latitude,
                "longitude": latlon.longitude,
                "altitude": latlon.altitude,
                "groundspeed": speed * 60,
                "track": latlon.bearing,
                "flight_id": task.flight_id,
            })
            
    def write_traffic_log(self,path:pathlib.Path) -> Traffic:
            traffic = Traffic(pd.DataFrame(self.__traffic_log))
            traffic.to_parquet(path)
            return traffic
    
    tempdir = tempfile.gettempdir()
    now = datetime.datetime.now()
    input_file = pathlib.Path(tempdir) / f"{now.strftime('%y-%m-%d_%H:%M:%S.%f')}_input.csv"
    output_file = pathlib.Path(tempdir) / f"{now.strftime('%y-%m-%d_%H:%M:%S.%f')}_output.json"
    
    def __setup_sim(self):
        assert self.tasklist is not None and len(self.tasklist) > 0, "No tasks to simulate"
        
        self.schedule_counters = dict()
        self.nopath_counter = dict()
        
        self.__t = self.tasklist[0].start_time
        self.__last_global_schedule = self.__t
        self.__end_of_times = max(t.end_time for t in self.tasklist)
        self.__task_index = dict()
        self.__last_schedules = dict()
        self.__schedules_count = 0
        self.__traffic_log = []
        
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
    
    def can_be_scheduled(self,task:FlightEndpoints,path:Path, backup:bool) -> bool:        
        # If it has a plan, don't schedule by default
        if task.planned:
            add_me = False
            
            # By default, do not reschedule already scheduled tasks
            if not backup:
                return False
            
            # Check if geometrically possible
            can_be_rescheduled = False
            # Reschedule only possible during a straight without incoming turn
            if path.sections[0].type == DubinsMove.STRAIGHT:
                if len(path.junctions) > 0:
                    if path.junctions[0] > task.stats.airspeed*self.reschedule_threshold.total_seconds()/60:
                        can_be_rescheduled = True
                else:
                    if path.total_length > task.stats.airspeed*self.reschedule_threshold.total_seconds()/60:
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
             threads:int=0) -> tuple[list[tuple[ACStats,Pose3D,float]],bool]:
        output:list[tuple[ACStats,Pose3D,float]] = [] # Aircraft stats, current pose, current speed
        
        
        ### Time forward
        new_t = timedelta + self.__t
        print("Time in simulation: ",new_t.strftime("%Y-%m-%d %H:%M:%S"))
        
        ### Metadata
        if self.influence_circle is not None and self.__xy_circle is None:
            self.__xy_circle = self.influence_circle.to_xy_NM(self.transformer)
        
        
        ## Check task to be done
        unplanned_task, new_paths, future_paths = self.collect_new_tasks(new_t, timedelta)
        
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
                output.append((s,p.start,p.speed_at_time(0)))
                
            # Delete almost ended plan
            if self.scheduled.duration < 1e-3:
                self.scheduled = None
                
                    
        self.__t = new_t
        

        loss_of_separation = False
        if len(output) > 1:
            if self.timedistance:
                min_dist, index1, index2 = min_tdist(self.separation, list((p, v) for _, p, v in output))
                loss_of_separation = min_dist < 0
            else:
                min_dist, index1, index2 = min_XY_dist(list(p for _,p,_ in output))
                loss_of_separation = min_dist < self.separation
            ac1 = output[index1][0].id
            ac2 = output[index2][0].id
            
            if loss_of_separation:
                print(f"LOSS OF SEPARATION: {min_dist:.2f} NM between {ac1} and {ac2}. ",end='')
                t1 = self.get_task(ac1)
                t2 = self.get_task(ac2)
                if t1.planned and t2.planned:
                    print(f"Both aircraft are planned!!!")
                    raise ValueError(f"LOSS OF SEPARATION: {min_dist:.2f} NM between {ac1} and {ac2}. Both aircraft are planned!!!")
                else:
                    if t1.planned:
                        t2.removed = True
                        t2.end_time = self.__t
                        if self.scheduled is not None:
                            del self.scheduled.trajectories[ac2]
                            self.__remove_from_axes(ac2)
                        print(f"Removing {ac2} from the simulation")
                    else:
                        t1.removed = True
                        t1.end_time = self.__t
                        if self.scheduled is not None:
                            del self.scheduled.trajectories[ac1]
                            self.__remove_from_axes(ac1)
                        print(f"Removing {ac1} from the simulation")
                    # input("Waiting for user to continue...")
                
                
        
        ## Scheduling rounds
        
        # Add for rescheduling those who can
        candidate_acs:set[int] = set()
        if self.scheduled is not None:
            for s,p in self.scheduled.trajectories.values():
                t = self.get_task(s.id)
                
                if self.can_be_scheduled(t,p,True):
                    candidate_acs.add(s.id)
        
        # Compute which aircraft can be rescheduled
        if self.__t - self.__last_global_schedule < self.reschedule_threshold and len(unplanned_task) == 0:
            candidate_acs.clear()
        
        # If some aircraft have to be rescheduled, do it
        success_schedule = True
        if len(candidate_acs) > 0:
            success_schedule = self.solve_path_planning(threads, new_t, future_paths, candidate_acs)
                        
        if self.__axes is not None:
            self.__update_axes(output, new_t, self.scheduled)
            self.__axes.set_title(str(new_t))
            plt.pause(0.001)
            
        return output, success_schedule

    def solve_path_planning(self, threads:int, new_t:pd.Timestamp, future_paths:dict[int,tuple[ACStats,Path,float]], candidate_acs:set[int]):
        ## Temporary files for the solver
        tempdir = tempfile.gettempdir()
        computer_now = datetime.datetime.now()
        input_file = pathlib.Path(tempdir) / f"{computer_now.strftime('%y-%m-%d_%H:%M:%S.%f')}_input.csv"
        output_file = pathlib.Path(tempdir) / f"{computer_now.strftime('%y-%m-%d_%H:%M:%S.%f')}_output.json"
        obstacles_file = pathlib.Path(tempdir) / f"{computer_now.strftime('%y-%m-%d_%H:%M:%S.%f')}_obstacles.json"
        success_schedule = True
        
        ## Check for aircraft to be considered as obstacles
        obstacles_exist = False
        noncandidate_acs = set()
        if self.scheduled is not None:
            print(f"Plan time: {self.scheduled.duration}")
            noncandidate_acs = set(self.scheduled.ids) - candidate_acs
                
            if len(noncandidate_acs) > 0:
                print_FleetPlan_to_JSON(obstacles_file,self.scheduled,noncandidate_acs,future_paths,True)
                
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
        write_pathplanning_problem_to_CSV(input_file,pp_problems,True)
            
        print(f"Solving problem including: {candidate_acs}\nObstacle ACs are: {noncandidate_acs}")
            
        ## Call the solver
        try:
            self.__schedules_count += 1
            solver_output = solve_problem(
                    self.solver_path,
                    input_file,
                    output_file,
                    self.separation,
                    self.timedistance,
                    (self.wind_x,self.wind_y),
                    threads,
                    obstacles_file if obstacles_exist else None,
                    self.geometric_obstacle_json_path if self.geometric_obstacle_json_path is not None and self.geometric_obstacle_json_path.exists() else None,
                    [self.cmd_shift],
                    [self.threshold_shift]
                )
                
            ## Parse the result and merge
            if solver_output.returncode == 0:
                solved = parse_trajectories_from_JSON(output_file)
                self.__last_global_schedule = self.__t
                for s,p in solved.trajectories.values():
                    if s.id in noncandidate_acs:
                        continue
                    t = self.get_task(s.id)
                    self.__last_schedules[s.id] = self.__t
                    self.schedule_counters[s.id] += 1
                    t.planned = True
                    
                
                if self.scheduled is None:
                    self.scheduled = solved
                else:
                    self.scheduled.merge(solved)
                    
                ## Update the task ends with the planning results
                for s,p in self.scheduled.trajectories.values():
                    t = self.get_task(s.id)
                    if t.planned:
                        path_duration = p.total_length/s.airspeed
                        dduration = (pd.Timedelta(minutes=path_duration) - (t.end_time - new_t)).total_seconds()/60
                        t.end_time = pd.Timedelta(minutes=path_duration) + new_t
                        dinit_duration = (t.end_time - t.initial_end_time).total_seconds()/60
                        if abs(dduration) > 0.1:
                            print(f"Change in arrival for {s.id}: {abs(dinit_duration):.1f} min {'earlier' if dduration < 0 else 'later'}")
                        
                        # assert poses_dist_2D(self.tasklist[i].end.to_pose3D(self.transformer,True),p.end) < 1e-3, f"Endpoints do not match for {s.id}: {self.tasklist[i].end.to_pose3D(self.transformer,True)} vs {p.end}"
                        # p_lat,p_lon = self.transformer.transform(p.end.x*NM_TO_METERS,p.end.y*NM_TO_METERS,direction='INVERSE')
                        # assert abs(p_lat-self.tasklist[i].end.latitude) < 1e-3, f"Latitudes do not match for {s.id}: {p_lat} vs {self.tasklist[i].end.latitude}"
                        # assert abs(p_lon-self.tasklist[i].end.longitude) < 1e-3, f"Longitudes do not match for {s.id}: {p_lon} vs {self.tasklist[i].end.longitude}"
                        
                    if self.__axes is not None:
                        poses = [p.pose_at_length(t) for t in np.linspace(0,p.total_length,50,endpoint=True)]
                            
                                
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
            raise e
        return success_schedule

    def collect_new_tasks(self, new_t:pd.Timestamp, timedelta:pd.Timedelta) -> tuple[set[int],dict[int,tuple[ACStats,Path]],dict[int,tuple[ACStats,Path,float]]]:
        """_summary_

        Args:
            new_t (pd.Timestamp): Current time
            timedelta (pd.Timedelta): Time step between previous and current time

        Returns:
            tuple[set[int],dict[int,tuple[ACStats,Path]],dict[int,tuple[ACStats,Path,float]]]: A tuple containing:
                - A set of unplanned task IDs
                - A dictionary containing tasks starting now with their default path (indexed by task ID)
                - A dictionary containing tasks starting in a near future with their default path (indexed by task ID)
        """
        unplanned_task:set[int] = set()
        new_paths:dict[int,tuple[ACStats,Path]] = {}
        future_paths:dict[int,tuple[ACStats,Path,float]] = {}
        for _,t in enumerate(self.tasklist):
            id = t.id
            
            # Task is removed due to loss of separation: skip it
            if t.removed:
                continue
            
            # Task already ended: remove drawing
            if t.end_time <= new_t:
                t.completed = True
                if (new_t - t.end_time) > timedelta*10:
                    self.__remove_from_axes(id)
                continue
            
            # Task yet to begin
            if t.start_time > new_t:
                # If the task is in the lookahead window, add it to the future paths
                if t.start_time - new_t <= self.lookahead_time:
                    start_pose = t.start.to_pose3D(self.transformer,True)
                    start_pose.z = 0
                    end_pose = copy.copy(start_pose)
                    end_pose.x += np.cos(start_pose.theta)*t.stats.airspeed*self.cmd_shift*10
                    end_pose.y += np.sin(start_pose.theta)*t.stats.airspeed*self.cmd_shift*10
                    path = Path.straight_path(start_pose,end_pose,t.stats.airspeed)
                    
                    future_paths[id] = (t.stats,path,(t.start_time-new_t).total_seconds()/60)
                    continue
                else: # Otherise, break (since tasks are sorted by increasing start time)
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
            end_pose.x += np.cos(start_pose.theta)*t.stats.airspeed*self.cmd_shift*3
            end_pose.y += np.sin(start_pose.theta)*t.stats.airspeed*self.cmd_shift*3
            path = Path.straight_path(start_pose,end_pose,t.stats.airspeed)
            
            new_paths[id] = (t.stats,path)
        return unplanned_task,new_paths,future_paths

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
                self.__log_traffic_states(states, self.__t)
            except KeyboardInterrupt:
                print("Simulation interrupted by user.")
                break
            
        plt.ioff()
        if self.__axes is not None:
            try:
                plot_endpoints_results(self.tasklist)
            except Exception as e:
                print("Could not produce report: ",e)
            
        return self.tasklist
        
    #################### Plotting ####################
        
    def attach_axes(self,ax:Axes,set_xylims:bool=True,runways_dict:dict[tuple[str,str,bool],ColorType]=dict()):
        self.__axes = ax
        self.__min_dist_line = ax.plot([],[],marker='D',linestyle=':',label=f"Min distance (-,-) : - NM",
                                       markeredgecolor='r',color='k')[0]
        
        # Setting up colors by destination
        
        runways_dict,self.flight_id_dict = make_colordict(self.tasklist,'tab10',10,runways_dict)   
        for t in self.tasklist:
            self.__color_dict[t.id] = self.flight_id_dict[t.flight_id]
        
        maxx = -np.inf
        maxy = -np.inf
        minx = np.inf
        miny = np.inf
        # Setting up runways colors
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
            
            
            color = self.__color_dict[t.id]
            if not t.departing:
                assert t.dest_airport is not None and t.dest_ICAO is not None
                if (t.dest_ICAO,t.dest_runway + " (arriving)") not in self.__dest_dict:
                    if t.dest_runway == "Unknown":
                        dest_line = self.__axes.plot(
                                    [],[],
                                    linestyle='-',
                                    alpha=0.5,
                                    color=color,
                                    label=f"{t.dest_ICAO} : {t.dest_runway} (arriving)"
                                )[0]
                    else:
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
                            label=f"{t.dest_ICAO} : {t.dest_runway} (arriving)"
                        )[0]
                    
                    self.__dest_dict[(t.dest_ICAO,t.dest_runway + " (arriving)")] = dest_line
            else:
                assert t.start_airport is not None and t.start_ICAO is not None
                if (t.start_ICAO,t.start_runway + " (departing)") not in self.__dest_dict:
                    if t.start_runway == "Unknown":
                        dest_line = self.__axes.plot(
                                    [],[],
                                    linestyle='-',
                                    alpha=0.5,
                                    color=color,
                                    label=f"{t.start_ICAO} : {t.start_runway} (departing)"
                                )[0]
                    else:
                        rw_1 = get_runway(t.start_airport,t.start_runway)
                        rw_2 = get_runway(t.start_airport,get_other_runway_name(t.start_runway))
                        
                        assert rw_1 is not None
                        assert rw_2 is not None
                        
                        proj1 = self.transformer.transform(rw_1.latitude,rw_1.longitude)
                        proj2 = self.transformer.transform(rw_2.latitude,rw_2.longitude)
                        
                        dest_line = self.__axes.plot(
                            [proj1[0]/NM_TO_METERS,proj2[0]/NM_TO_METERS],[proj1[1]/NM_TO_METERS,proj2[1]/NM_TO_METERS],
                            linestyle='-',
                            alpha=0.5,
                            color=color,
                            label=f"{t.start_ICAO} : {t.start_runway} (departing)"
                        )[0]
                    
                    self.__dest_dict[(t.start_ICAO,t.start_runway + " (departing)")] = dest_line
        
        if set_xylims:
            self.__axes.set_xlim(minx,maxx)
            self.__axes.set_ylim(miny,maxy)
        
        self.__axes.set_xlabel(f"Easting (NM, {self.transformer.target_crs.name})")
        self.__axes.set_ylabel(f"Northing (NM, {self.transformer.target_crs.name})")
        
        self.__axes.legend()
        
        return runways_dict
        
    def __remove_from_axes(self,ac_id:int):    
        try:
            self.__pos_dict[ac_id].set_visible(False)
            self.__pos_dict[ac_id].remove()
            del self.__pos_dict[ac_id]
        except KeyError:
            pass
        
        try:
            self.__line_dict[ac_id].set_visible(False)
            self.__line_dict[ac_id].remove()
            del self.__line_dict[ac_id]
        except KeyError:
            pass
        
        try:
            self.__quiver_dict[ac_id].set_visible(False)
            self.__quiver_dict[ac_id].remove()
            del self.__quiver_dict[ac_id]
        except KeyError:
            pass
        
        try:
            self.__label_dict[ac_id].set_visible(False)
            self.__label_dict[ac_id].remove()
            del self.__label_dict[ac_id]
        except KeyError:
            pass
        
        try:
            self.__traj_dict[ac_id].set_visible(False)
            self.__traj_dict[ac_id].remove()
            del self.__traj_dict[ac_id]
        except KeyError:
            pass
        
    def __update_axes(self, states:list[tuple[ACStats,Pose3D,float]], new_t:pd.Timestamp, trajs:Optional[FleetPlan]=None) -> Optional[tuple[float,int,int]]:
        """
        Update the axes given the input states and time.

        Args:
            states (list[tuple[ACStats,Pose3D,float]]): A list of tuples containing the aircraft stats, pose, and speed.
            new_t (pd.Timestamp): The current time.

        Returns:
            Optional[tuple[float,int,int]]: If there are at least two states, returns the minimum distance and the IDs of the two aircraft that are closest. Otherwise, returns None.
        """
        if self.__axes is None:
            return
        
        for s,p,_ in states:
            id = s.id
            
            task = self.tasklist[self.__task_index[id]]
            dest_pose = task.end.to_pose3D(self.transformer,True)
            ## Clean trajs 
            try:
                self.__traj_dict[s.id].set_visible(False)
                self.__traj_dict[s.id].remove()
                del self.__traj_dict[s.id]
            except KeyError:
                pass
            
            if trajs is not None:
                try:
                    _,path = trajs.get_path(s.id)
                    poses = [path.pose_at_length(t) for t in np.linspace(0,path.total_length,50,endpoint=True)]
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
                endpoint = self.__axes.plot([p.x,dest_pose.x],[p.y,dest_pose.y],linestyle="",marker='^',markerfacecolor=(0,0,0,0),markeredgecolor=color)[0]
                self.__pos_dict[id] = endpoint
            
            if not self.is_scheduled(id):
                endpoint.set_marker('P')
            else:
                endpoint.set_marker('^')
            endpoint.set_xdata([p.x,dest_pose.x])
            endpoint.set_ydata([p.y,dest_pose.y])
                
                
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
            if self.timedistance:
                min_dist, index1, index2 = min_tdist(self.separation, list((p, v) for _, p, v in states))
                loss_of_separation = min_dist < 0
            else:
                min_dist, index1, index2 = min_XY_dist(list(p for _,p,_ in states))
                loss_of_separation = min_dist < self.separation
                
            stat1,p1,v1 = states[index1]
            stat2,p2,v2 = states[index2]
            
            if self.timedistance:
                actual_dist = poses_XY_dist(p1,p2)
                max_speed = max(v1,v2)
                label=f"Min slack    ({stat1.id},{stat2.id}) : {min_dist:.2f} NM\nActual dist is {actual_dist:.2f} NM for time {self.separation} min and speed {max_speed:.2f} NM/min ({max_speed*60:.2f} kts)"
            else:
                label=f"Min distance ({stat1.id},{stat2.id}) : {min_dist:.2f} NM"
            
            
            if loss_of_separation:
                label += "\n!!! LOSS OF SEPARATION !!!"
                
            self.__min_dist_line.set_xdata([p1.x,p2.x])
            self.__min_dist_line.set_ydata([p1.y,p2.y])
            self.__min_dist_line.set_label(label)
                
            self.__axes.legend(*self.__axes.get_legend_handles_labels())
            
            return min_dist,stat1.id,stat2.id
        else:
            return None        

#################### Entrypoint ####################


def main():
    import argparse,cProfile
    
    parser = argparse.ArgumentParser(
        'Landing scheduler'
    )
    parser.add_argument('solver',help='Location of the DubinsFleetPlanner solver')
    
    parser.add_argument('-d','--mdist',type=float,help='Minimal distance separating aircraft, in NM. Default to 3.',default=3)
    parser.add_argument('--lspeed',type=float,help='Desired final speed for landing aircraft, in knots. Default to 100.',default=200)
    parser.add_argument('--dspeed',type=float,help='Desired final speed for departing aircraft, in knots. Default to 400.',default=500)
    parser.add_argument('--cspeed',action='store_true',help="Flag. If set, airspeed is set to a constant (the target value)")
    parser.add_argument('-r','--turn-radius',dest='turn_radius',
        type=float, help='Minimal turn radius for all aircraft. Default to 1.33 NM (full turn in 2 minutes at 250 kt)',default=1.33)
    # parser.add_argument('-w','--wind',dest='wind', nargs=2,
    #                     help='XY Wind, as a pair of values (X,Y). Default to (0,0), i.e. no wind.',
    #                     default=(0,0))
    parser.add_argument('--tdist',dest='tdist',action='store_true',help='If set, interpret mdist as time-based separation (in minutes). Default to False.',default=False)
    parser.add_argument('--data',help='A Traffic-compatible data file. If not set, use the sample dataset of Traffic',default=None)
    parser.add_argument('--period',help="Period duration for splitting the input data into subproblems (e.g., 'D' for daily, 'W' for weekly). Full details at https://pandas.pydata.org/docs/user_guide/timeseries.html#timeseries-offset-aliases.",
                        default='D')
    parser.add_argument('--ICAOs',nargs='+',
                        help="List of ICAO codes for airports in which to look for landings in the given dataset. Default to LFPO,LFPG,LFPB",
                        default=["LFPO","LFPG","LFPB"])
    parser.add_argument('-s','--threshold-shift',type=float,dest='threshold_shift',
                        help='Distance relative to runway threshold for defining last straight, in NM. Default to 10.',default=10)
    parser.add_argument('--cmd-shift',type=float,dest='cmd_shift',
                        help="Duration (in minutes) for which no commands should issued (that is, go straight). Default to 1 minute.", default=1)
    parser.add_argument('--lookahead',type=float,dest='lookahead',default=30,
                        help="Look ahead duration (in minutes). Describe how far in the future should one look ahead before planning new trajectories in order to take into accounts new aircraft.")
    parser.add_argument('--influence-radius',type=float,dest='influence_radius',
                        help="Radius of influence from the airports barycenter, in NM. If set, aircraft will be scheduled only within this radius. Disabled by default.",default=None)
    parser.add_argument('--departing-radius',type=float,dest='departing_radius',
                        help="Influence radius for departing aircraft. Only relevant if 'influence-radius' is already set. Modified value applied to departing aircraft. Default to be equalt to 'influence-radius'.",default=None)
    parser.add_argument('--epsg',type=int,help="EPSG code for XY projection of (lat,lon) coordinates from WGS84. Default to the UTM zone of the first airport in the ICAO list.",
                        default=None)
    
    parser.add_argument('-ts','--timestep',type=float,
                        help="Simulation time step, in minutes. Default to 1/6 (ie 10s).",default=1/6)
    parser.add_argument('-I','--intervals',nargs=3,
                        help="Triplet (min,max,step) defining the offsets to the reference time for the target times. Defined in minutes. Default to (-10,10,2)",
                        default=(-10,10,2))
    parser.add_argument('-m','--max-reschedule',dest="max_reschedule",type=int,
                        help="Maximum number of times a flight can be rescheduled. Default to 5.",default=5)
    parser.add_argument('-t','--time-threshold',dest='time_threshold',type=float,
                        help="Duration ellapsed after which we reschedule aircraft. Default to 5 minutes.",default=5)
    parser.add_argument('-f','--final-time',dest='final_time',type=float,
                        help="Final duration in minutes. If an aircraft is less than this duration away from landing, it cannot be rescheduled. Default to 5 minutes.",
                        default=5)
    parser.add_argument('--threads',dest="threads",type=int,
                        help="Number of threads to be used by the solver. 0 allows it to autoselect. Default to 0.",default=0)
    parser.add_argument('--profile',dest="profile",action='store_true',
                        help="If set, profile the simulation and print the results at the end. Default to False.",default=False)
    parser.add_argument('--data-output',type=pathlib.Path,default=None,dest="data_output",
                        help="Write simulation data to this folder (created if missing). Two files per period will be created: a parquet file for the traffic data and a Python pickle file for the task data and used arguments.")
    parser.add_argument('-ui',dest="set_ui",action='store_true',
                        help="If set, display the simulation UI.")
    args = parser.parse_args()
    
    solver = args.solver
    print("===== Parsing traffic... =====\n")
    if args.data is None:
        traffic_iter = [samples.quickstart]
    else:
        traffic_iter = iter_flight_by_period(args.data,'2h',id_column='flight_id')
        if traffic_iter is None:
            print("ERROR: Importing traffic failed. Exiting")
            exit(1)
    
    airports = get_airports(args.ICAOs)
    if args.epsg is None:
        transformer = get_airport_latlon_transformer(airports[0])
    else:
        transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{args.epsg}")

    expected_landing_speed = args.lspeed # kts
    expected_departing_speed = args.dspeed # kts
    threshold_shift = args.threshold_shift # NM
    timeshifts = pd.timedelta_range(f"{args.intervals[0]} minute",f"{args.intervals[1]} minute",freq=f"{float(args.intervals[2])*60}s").to_list()

    ## Generate borders ##
    geometric_obstacle_json_path = None
    influence_circle = None
    outgoing_circle = None
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
        
        if args.departing_radius is not None:
            outgoing_circle = InfluenceCircle.from_xy_meters(avgx,avgy,args.departing_radius*NM_TO_METERS,transformer)
        else:
            outgoing_circle = influence_circle
    
    ## Plotting 
    runway_colordict = dict()

    for traffic in traffic_iter:
        print(f"Processing traffic from {traffic.data['timestamp'].min()} to {traffic.data['timestamp'].max()}")
        filtered_traffic = filter_traffic(traffic,airports)
        if filtered_traffic is None:
            print("No flights found for the given ICAO codes. Skipping to next period.")
            continue
        print(filtered_traffic)
        print(f"Total flights to process: {len(filtered_traffic.flight_ids)}")
        
        if args.set_ui:
            plt.ion()
            fig,ax = plt.subplots(figsize=(16/1.5,9/1.5))
            ax.set_aspect('equal')
            fig.tight_layout()
            plt.show()
        
        endpoints:dict[str,FlightEndpoints] = generate_flightEndpoints(airports, expected_landing_speed, expected_departing_speed, filtered_traffic, args.cspeed, outgoing_circle)
        print(f"Total flight endpoints generated: {len(endpoints)}")
        # for ep in endpoints:
            # ax.scatter(ep.start.to_pose3D(transformer,True).x,ep.start.to_pose3D(transformer,True).y,marker='o',color='k',alpha=0.2)
            # ax.scatter(ep.end.to_pose3D(transformer,True).x,ep.end.to_pose3D(transformer,True).y,marker='x',color='k',alpha=0.2)
        
        print("\n===== Traffic parsing done! =====\nSetting up simulator...")
        
        sim = ArrivalsSimulator(pathlib.Path(solver),
                                transformer,
                                timeshifts,
                                threshold_shift,
                                args.cmd_shift*expected_landing_speed/60,
                                pd.Timedelta(f"{args.time_threshold} minute"),
                                pd.Timedelta(f"{args.final_time} minute"),
                                max_reschedule=args.max_reschedule,
                                separation=args.mdist)
        
        sim.timedistance = args.tdist
        sim.lookahead_time = pd.Timedelta(minutes=args.lookahead)
        
        if geometric_obstacle_json_path is not None:
            sim.geometric_obstacle_json_path = geometric_obstacle_json_path
        if influence_circle is not None:
            sim.influence_circle = influence_circle
            x,y,_ = influence_circle.to_xy_NM(transformer)
            # The plot uses NM, it is easier for distance comparison.
            if args.set_ui:
                ax.add_patch(
                    Circle((x,y),radius=influence_circle.radius,fill=False,linestyle='--',color='k',label=f"Influence radius ({influence_circle.radius:.1f} NM)")
                )
                if outgoing_circle is not None and abs(outgoing_circle.radius - influence_circle.radius) > 1e-2:
                    xbis,ybis,_ = outgoing_circle.to_xy_NM(transformer)
                    
                    ax.add_patch(
                        Circle((xbis,ybis),radius=outgoing_circle.radius,fill=False,linestyle='--',color='g',label=f"Departing radius ({outgoing_circle.radius:.1f} NM)")
                    )
        
        ## Run simulation ##
        
        sim.setup_simulation(list(endpoints.values()))
        
        # If UI is enabled, attach the axes to the simulation
        if args.set_ui:
            runway_colordict = sim.attach_axes(ax,runways_dict=runway_colordict)
        
        start_time = time()
        
        if args.profile:
            print("Profiling simulation...")
            cProfile.runctx('sim.simulate(pd.Timedelta(f"{args.timestep} minute"), args.threads)', globals(), locals(),filename='simulation_profile.prof')
        else:
            task_results = sim.simulate(pd.Timedelta(f"{args.timestep} minute"), args.threads)
            if args.data_output is not None:
                data_output = pathlib.Path(args.data_output)
                data_output.mkdir(parents=True,exist_ok=True)
                
                sim.write_traffic_log(data_output/f"simulated_traffic_{'-'.join(args.ICAOs)}_{traffic.data['timestamp'].min().strftime('%Y%m%d_%H%M')}_{traffic.data['timestamp'].max().strftime('%Y%m%d_%H%M')}.parquet")
                summary = SimulationSummary(task_results,args)
                summary_path = data_output/f"simulation_summary_{traffic.data['timestamp'].min().strftime('%Y%m%d_%H%M')}_{traffic.data['timestamp'].max().strftime('%Y%m%d_%H%M')}.pkl"
                with open(summary_path,'wb') as f:
                    pickle.dump(summary,f)
        
        end_time = time()
        print(f"Simulation time: {end_time - start_time:.2f} seconds. Called solver {sim.schedules_count} times.")
        cont = input("Enter 'y' to start simulation with the next period, quit otherwise: ")
        if cont != 'y':
            break
    
if __name__ == '__main__':
    main()
