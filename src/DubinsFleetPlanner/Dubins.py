# Copyright (C) 2025 Mael FEURGARD <mael.feurgard@enac.fr>
# 
# This file is part of DubinsFleetPlanner.
# 
# DubinsFleetPlanner is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# DubinsFleetPlanner is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with DubinsFleetPlanner.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations
import typing

from dataclasses import dataclass,field,asdict
from enum import IntEnum
import copy

import numpy as np

from .Poses import Pose2D,Pose3D,ListOfTimedPoses,poses_euclidean_dist
from .Aircraft import ACStats
from .SpeedProfile import SpeedProfile

#################### Base elements ####################

class DubinsMove(IntEnum):
    STRAIGHT = 0
    LEFT     = 1
    RIGHT    = 2
    
    def __str__(self) -> str:
        return self.name
    
    def abbr(self) -> str:
        return self.name[0]


################ Base Dubins generation ################

def mod2pi(theta:float) -> float:
    return np.mod(theta, 2.0 * np.pi)

def central_angle(theta:float) -> float:
    output = np.fmod(theta, 2*np.pi)
    
    if output > np.pi:
        return output - 2*np.pi
    
    if output < -np.pi:
        return output + 2*np.pi
    
    return output

def transform(start:Pose2D, goal:Pose2D) -> tuple[float,float,float]:
    x0, y0, th0 = start
    x1, y1, th1 = goal

    dx = x1 - x0
    dy = y1 - y0

    phi = np.arctan2(dy, dx)
    d = np.hypot(dx, dy)

    alpha = mod2pi(th0 - phi)
    beta  = mod2pi(th1 - phi)

    return d, alpha, beta


def LSL(start:Pose2D, goal:Pose2D) -> typing.Optional[tuple[float,float,float]]:
    d, a, b = transform(start, goal)

    sa, sb = np.sin(a), np.sin(b)
    ca, cb = np.cos(a), np.cos(b)

    tmp = d**2 + 2 - 2*np.cos(a - b) + 2*d*(sa - sb)
    if tmp < 0:
        return None

    u = np.sqrt(tmp)
    theta = np.arctan2(cb - ca, d + sa - sb)

    t = mod2pi(theta - a)
    v = mod2pi(b - theta)

    return t, u, v

def RSR(start:Pose2D, goal:Pose2D) -> typing.Optional[tuple[float,float,float]]:
    d, a, b = transform(start, goal)

    sa, sb = np.sin(a), np.sin(b)
    ca, cb = np.cos(a), np.cos(b)

    tmp = d**2 + 2 - 2*np.cos(a - b) + 2*d*(sb - sa)
    if tmp < 0:
        return None

    u = np.sqrt(tmp)
    theta = np.arctan2(ca - cb, d - sa + sb)

    t = mod2pi(a - theta)
    v = mod2pi(theta - b)

    return t, u, v

def LSR(start:Pose2D, goal:Pose2D) -> typing.Optional[tuple[float,float,float]]:
    d, a, b = transform(start, goal)

    sa, sb = np.sin(a), np.sin(b)
    ca, cb = np.cos(a), np.cos(b)

    tmp = d**2 - 2 + 2*np.cos(a - b) + 2*d*(sa + sb)
    if tmp < 0:
        return None

    u = np.sqrt(tmp)
    theta = np.arctan2(-ca - cb, d + sa + sb)

    t = mod2pi(theta - a - np.arctan2(-2,u))
    v = mod2pi(theta - b - np.arctan2(-2,u))

    return t, u, v

def RSL(start:Pose2D, goal:Pose2D) -> typing.Optional[tuple[float,float,float]]:
    d, a, b = transform(start, goal)

    sa, sb = np.sin(a), np.sin(b)
    ca, cb = np.cos(a), np.cos(b)

    tmp = d**2 - 2 + 2*np.cos(a - b) - 2*d*(sa + sb)
    if tmp < 0:
        return None

    u = np.sqrt(tmp)
    theta = np.arctan2(ca + cb, d - sa - sb)

    t = mod2pi(a - theta + np.arctan2(2,u))
    v = mod2pi(b - theta + np.arctan2(2,u))

    return t, u, v

def RLR(start:Pose2D, goal:Pose2D) -> typing.Optional[tuple[float,float,float]]:
    d, a, b = transform(start, goal)

    sa, sb = np.sin(a), np.sin(b)
    ca, cb = np.cos(a), np.cos(b)

    tmp = (6 - d**2 + 2*np.cos(a - b) + 2*d*(sa - sb)) / 8
    if np.abs(tmp) > 1:
        return None

    u = mod2pi(2*np.pi - np.arccos(tmp))
    theta = np.arctan2(ca - cb, d - sa + sb)

    t = mod2pi(a - theta + u / 2)
    v = mod2pi(a - b - t + u)

    return t, u, v

def LRL(start:Pose2D, goal:Pose2D) -> typing.Optional[tuple[float,float,float]]:
    d, a, b = transform(start, goal)

    sa, sb = np.sin(a), np.sin(b)
    ca, cb = np.cos(a), np.cos(b)

    tmp = (6 - d**2 + 2*np.cos(a - b) + 2*d*(sb - sa)) / 8
    if np.abs(tmp) > 1:
        return None

    u = mod2pi(2*np.pi - np.arccos(tmp))
    theta = np.arctan2(ca - cb, d + sa - sb)

    t = mod2pi(-a - theta + u / 2)
    v = mod2pi(b - a - t + u)

    return t, u, v

def SRS(start:Pose2D, goal:Pose2D) -> typing.Optional[tuple[float,float,float]]:
    d, a, b = transform(start, goal)
    
    da = central_angle(a - b)
    if np.isclose(da,0.0):
        return None
    
    cs = d * abs(np.sin(b)/np.sin(da))
    ec = d * abs(np.sin(a)/np.sin(da))
    
    
    first_l = cs - np.tan(da/2)
    second_l = mod2pi(a - b)
    third_l = ec - np.tan(da/2)
    
    if first_l < 0 or third_l < 0:
        return None
    else:
        return first_l,second_l,third_l
    
def SLS(start:Pose2D, goal:Pose2D) -> typing.Optional[tuple[float,float,float]]:
    d, a, b = transform(start, goal)
    # Trajectory goes from S=(0,0) to E=(d,0)
    # with start angle a and end angle b.
    # Start and end axes intersect at point C
    
    da = central_angle(b - a)
    if np.isclose(da,0.0):
        return None
    
    cs = d * np.abs(np.sin(b)/np.sin(da)) # Law of sines on triangle SCE
    ec = d * np.abs(np.sin(a)/np.sin(da)) # to compute SC and CE

    first_l = cs - np.tan(da/2)
    second_l = mod2pi(b - a)
    third_l = ec - np.tan(da/2)
    
    if first_l < 0 or third_l < 0:
        return None
    else:
        return first_l,second_l,third_l
    
    
def all_paths(start:Pose2D, goal:Pose2D) -> tuple[\
                                typing.Optional[tuple[float,float,float]],\
                                typing.Optional[tuple[float,float,float]],\
                                typing.Optional[tuple[float,float,float]],\
                                typing.Optional[tuple[float,float,float]],\
                                typing.Optional[tuple[float,float,float]],\
                                typing.Optional[tuple[float,float,float]],\
                                typing.Optional[tuple[float,float,float]],\
                                typing.Optional[tuple[float,float,float]]]:
    """Returns paths component lengths (or None is solution does not exists) for
    LSL, RSR, LSR, RSL, RLR, LRL, SRS, SLS in that order
    """

    return (LSL(start,goal),
            RSR(start,goal),
            LSR(start,goal),
            RSL(start,goal),
            RLR(start,goal),
            LRL(start,goal),
            SRS(start,goal),
            SLS(start,goal),)

#################### Paths elements ####################

@dataclass
class BasicPath:
    type  : DubinsMove # Either LEFT,RIGHT or STRAIGHT, for describing a turn or a straight line movement,
    length: float # Length of this section
    x     : float # X coordinate of the refence point; For a Straight: Starting point | For a turn: Circle center
    y     : float # Y coordinate of the refence point
    z     : float # Z coordinate of the refence point
    p1    : float # For a Straight: horizontal x speed | For a turn: Radius
    p2    : float # For a Straight: horizontal y speed | For a turn: signed angular speed; positive when LEFT, negative when RIGHT
    p3    : float # Vertical speed
    p4    : float # For a Straight: unused | For a turn: initial angle
    
    def __post_init__(self):
        if type(self.type) is str:
            if self.type == "STRAIGHT":
                self.type = DubinsMove.STRAIGHT
            elif self.type == "LEFT":
                self.type = DubinsMove.LEFT
            elif self.type == "RIGHT":
                self.type = DubinsMove.RIGHT
            else:
                raise ValueError(f"Unknown Dubins Move:\n{self.type}")
        elif type(self.type) is not DubinsMove:
            self.type = DubinsMove(self.type)

        if self.type == DubinsMove.STRAIGHT:
            norm = np.hypot(self.p1, self.p2)
            if np.isclose(norm, 0.):
                raise ValueError("A straight path must have non-zero horizontal speed")
            self.p1 /= norm
            self.p2 /= norm
        else:
            if np.isclose(self.p1, 0.) or np.isclose(self.p2, 0.):
                raise ValueError("A turn path must have non-zero radius and angular speed")
            self.p2 = np.sign(self.p2) / self.p1
                        
    ##### Extra constructors #####
        
    @staticmethod
    def from_2D(type: DubinsMove, length: float, start:Pose2D, radius:float) -> BasicPath:
        dx = np.cos(start.angle)
        dy = np.sin(start.angle)
        
        if type is DubinsMove.STRAIGHT:
            return BasicPath(type,length,
                             start.x,start.y,0.,
                             dx,dy,
                             0.,0.)
            
        else:
            p1 = radius
            p2 = 1/radius
            
            if type is DubinsMove.LEFT:
                cx = start.x - dy*radius
                cy = start.y + dx*radius
            else:
                cx = start.x + dy*radius
                cy = start.y - dx*radius
                p2 *= -1
            
            p4 = np.arctan2(start.y - cy, start.x - cx)
            return BasicPath(type,length,
                             cx,cy,0.,
                             p1,p2,0.,p4)
            
    @staticmethod
    def from_3D(type: DubinsMove, length: float, start:Pose3D, radius:float) -> BasicPath:
        start_2d = start.to2D()
        result = BasicPath.from_2D(type,length,start_2d,radius)
        result.z = start.z
        result.p3 = 0.
        return result
    
    @staticmethod
    def from_2_points(point0:tuple[float,float,float],point1:tuple[float,float,float]) -> BasicPath:
        t = DubinsMove.STRAIGHT
        length = np.sqrt((point0[0]-point1[0])**2 + (point0[1]-point1[1])**2 + (point0[2]-point1[2])**2)
        x = point0[0]
        y = point0[1]
        z = point0[2]
        p1 = point1[0]-point0[0]
        p2 = point1[1]-point0[1]
        p3 = point1[2]-point0[2]
        p4 = 0
        return BasicPath(t,length,x,y,z,p1,p2,p3,p4)
    
    @staticmethod
    def from_circle(cx:float,cy:float,radius:float, cz:float=0.) -> BasicPath:
        t = DubinsMove.LEFT
        length = 2*np.pi*radius
        x = cx
        y = cy
        z = cz
        p1 = radius
        p2 = 1/radius
        p3 = 0.
        p4 = 0.
        return BasicPath(t,length,x,y,z,p1,p2,p3,p4)
    
    ##### Getters #####
    def speed_XY(self) -> float:
        if self.type == DubinsMove.STRAIGHT:
            return np.hypot(self.p1,self.p2)
        else:
            return self.p1*np.abs(self.p2)
        
    def set_XY_speed(self,speed:float):
        if self.type == DubinsMove.STRAIGHT:
            norm = np.hypot(self.p1,self.p2)
            self.p1 = speed*self.p1/norm
            self.p2 = speed*self.p2/norm
        else:
            self.p2 = np.sign(self.p2)*speed/self.p1

    def radius(self) -> float:
        if self.type == DubinsMove.STRAIGHT:
            return 0.
        else:
            return self.p1
        
    def pose_at_length(self,length:float) -> Pose3D:
        # assert np.isclose(self.p3,0.)
        z = self.z + length*self.p3
        
        speed = self.speed_XY()
        
        if self.type == DubinsMove.STRAIGHT:
            x = self.x + length*self.p1/speed
            y = self.y + length*self.p2/speed
            angle = np.arctan2(self.p2,self.p1)
            return Pose3D(x,y,z,angle)
        else:
            center = np.array([self.x,self.y])
            a = self.p4 + np.sign(self.p2)*length/self.p1
            
            xy_pos = center + self.p1*np.array([np.cos(a),np.sin(a)])
            
            angle = a + (np.pi/2 if self.type == DubinsMove.LEFT else -np.pi/2)
            return Pose3D(xy_pos[0],xy_pos[1],z,angle)
        
    def start(self) -> Pose3D:
        return self.pose_at_length(0)
    
    def end(self) -> Pose3D:
        return self.pose_at_length(self.length)
    
    ##### Export #####
    
    def asdict(self) -> dict[str,float|DubinsMove]:
        output = asdict(self)
        output["type"] = str(output["type"])
        output["m"] = self.type.value
        return output
    
@dataclass
class Path:
    total_length:float  # Total length of the path
    start       :Pose3D # Starting pose
    end         :Pose3D # Ending pose
    sections    :list[BasicPath]    # List of base primitives describing the shape
    junctions   :list[float] = field(init=False) # List of lengths at which there is a section change
    speed_profile:SpeedProfile = field(default_factory=SpeedProfile) # Speed profile along the path
    
    @staticmethod
    def from_sections_only(sections:list[BasicPath]) -> Path:
        total_length = sum(s.length for s in sections)
        start = sections[0].start()
        end = sections[-1].end()
        return Path(total_length,start,end,sections)
    
    @staticmethod
    def __compute_junctions(sections:list[BasicPath]) -> list[float]:
        output = []
        last_length = 0.
        for s in sections[:-1]:
            last_length += s.length
            output.append(last_length)
        return output
    
    @staticmethod
    def straight_path(start:Pose3D,end:Pose3D) -> Path:
        length = poses_euclidean_dist(start,end)
        section = BasicPath.from_2_points((start.x,start.y,start.z),(end.x,end.y,end.z))
        return Path(length,start,end,[section])
    
    def __post_init__(self):
        assert np.isclose(self.total_length, sum(s.length for s in self.sections),1e-4)
        self.junctions = self.__compute_junctions(self.sections)
        for i in range(1,len(self.sections)):
            assert np.all(np.isclose(self.sections[i-1].end().astuple(),self.sections[i].start().astuple(),1e-4))
    
    def speed_at_time(self,time:float) -> float:
        return self.speed_profile.get_speed_at_time(time)
    
    def length_at_time(self,time:float) -> float:
        return self.speed_profile.get_length_at_time(time)
    
    def time_at_length(self,length:float) -> float:
        return self.speed_profile.get_time_at_length(length)
    
    def duration(self) -> float:
            return self.time_at_length(self.total_length)
    
    def max_turn_radius(self) -> float:
        output = 0.
        for s in self.sections:
            output = max(abs(s.radius()),output)
        return output
        
    def shift(self,dx:float,dy:float,dz:float=0.):
        self.start.x += dx
        self.start.y += dy
        self.start.z += dz
        
        self.end.x += dx
        self.end.y += dy
        self.end.z += dz
        
        for s in self.sections:
            s.x += dx
            s.y += dy
            s.z += dz
    
    def pose_at_time(self,time:float) -> Pose3D:
        return self.pose_at_length(self.speed_profile.get_length_at_time(time))

    def poses_at_times(self,times:np.ndarray) -> list[Pose3D]:
        # Check ts is sorted
        assert np.all(times[:-1] <= times[1:])
        
        lengths_traveled = np.fromiter((self.speed_profile.get_length_at_time(t) for t in times),
                                       float,len(times))
        
        return self.poses_at_lengths(lengths_traveled)

    def pose_at_length(self,length:float) -> Pose3D:
        section_id = 0
        if length > self.total_length:
            return Pose3D.undefined()
        
        while section_id < len(self.junctions) and length > self.junctions[section_id]:
            length -= self.sections[section_id].length
            section_id += 1
        
        if (section_id >= len(self.junctions)):
            return self.sections[-1].pose_at_length(length)
        else:
            return self.sections[section_id].pose_at_length(length)
    
    def poses_at_lengths(self,lengths:np.ndarray) -> list[Pose3D]:
        # Check ts is sorted
        assert np.all(lengths[:-1] <= lengths[1:])
        
        section_id = 0
        output = []
        prev_sections_time = 0
        
        for time in lengths:
            if time > self.total_length:
                return output
            
            while section_id < len(self.junctions) and time > self.junctions[section_id]:
                prev_sections_time += self.sections[section_id].length
                section_id += 1
            
            if (section_id >= len(self.junctions)):
                output.append(self.sections[-1].pose_at_length(time-prev_sections_time))
            else:
                output.append(self.sections[section_id].pose_at_length(time-prev_sections_time))
                
        return output
    
    def sample(self,num:int) -> list[Pose3D]:
        assert num > 0
        return self.poses_at_lengths(np.linspace(0,self.total_length,num))
    
    def section_at(self,length:float) -> typing.Optional[BasicPath]:
        section_id = 0
        time = length
        if time > self.total_length:
            return None
        
        while section_id < len(self.junctions) and length > self.junctions[section_id]:
            time -= self.sections[section_id].length
            section_id += 1
        
        if (section_id >= len(self.junctions)):
            return self.sections[-1]
        else:
            return self.sections[section_id]
    
    def asdict(self) -> dict[str,typing.Any]:
        output = asdict(self)
        output.pop("junctions")
        output["sections_count"] = len(self.sections)
        sections = []
        for s in self.sections:
            sections.append(s.asdict())
        output["sections"] = sections
        return output
    
    def join(self,other:Path):
        s_length = self.total_length
        self.total_length += other.total_length
        self.end = other.end

        self.junctions.append(s_length)
        
        o_junctions = other.junctions.copy()
        for i in range(len(o_junctions)):
            o_junctions[i] += s_length
        self.junctions.extend(o_junctions)
        
        self.sections.extend(other.sections)
        
        
    def abbr(self) -> str:
        return ''.join(s.type.abbr() for s in self.sections)    
            
    
    def follow_for(self,length:float) -> typing.Optional[Path]:
        section_id = 0
        remaining_length = length
        if remaining_length > self.total_length:
            return None
        
        while section_id < len(self.junctions) and length > self.junctions[section_id]:
            remaining_length -= self.sections[section_id].length
            section_id += 1
            
        new_length = self.total_length - length
        new_end     = self.end
        
        if (section_id >= len(self.junctions)):
            new_sections = [self.sections[-1]]
        else:
            new_sections = self.sections[section_id:]
            
        new_start   = new_sections[0].pose_at_length(remaining_length)
        
        new_start_section = BasicPath.from_3D(
            new_sections[0].type,
            new_sections[0].length-remaining_length,
            Pose3D(new_start.x,new_start.y,new_start.z,new_start.theta),
            new_sections[0].radius()
        )
        
        if new_sections[0].type == DubinsMove.STRAIGHT:
            dtheta = np.mod(new_start.theta-new_start_section.end().theta,2*np.pi)
            assert np.isclose(dtheta,0.,1e-4) or np.isclose(dtheta,2*np.pi,1e-4), f"New start section end angle {new_start_section.end().theta} does not match new start angle {new_start.theta} : value: {dtheta}"

        new_sections[0] = new_start_section
        new_length = sum(s.length for s in new_sections)
        
        output = Path(new_length,new_start,new_end,copy.deepcopy(new_sections))
        
        assert np.isclose(output.total_length,new_length,1e-4)
        
        return output

def path_extra_length(p:list[BasicPath]|Path) -> float:
    """ Return the 'extra length' used by straight-extended paths (4 or 5 components)
    """
    if isinstance(p,Path):
        return path_extra_length(p.sections)
    
    abbr = ''.join(e.type.abbr() for e in p)
    if len(abbr) <= 3:
        return 0.
    else:
        if len(abbr) > 5:
            raise AttributeError("Cannot handle path with that many sections!!: "+abbr)
        
        if len(abbr) == 5 and abbr[0] == 'S' and abbr[-1] == 'S':
            return p[0].length + p[4].length
        else: #len(abbr) == 4
            if abbr[0] == 'S' and abbr[-1] == 'S': # Straight extended + SCS
                if abbr[1] == 'S':
                    return p[0].length
                else:
                    return p[-1].length
            else: # Straight extended + classic Dubins
                if abbr[0] == 'S':
                    return p[0].length
                elif abbr[-1] == 'S':
                    return p[-1].length
                else:
                    raise AttributeError("Unknown straight extended path: "+abbr)
    

@dataclass
class FleetPlan:
    separation  :float # Minimal separation achieved
    z_alpha     :float # Z distance modifier value
    wind_x      :float # Wind speed along the X axis 
    wind_y      :float # Wind speed along the Y axis
    duration    :float # Time duration of the plan
    trajectories:dict[int,tuple[ACStats,Path]] # List of paths and stats
            
    def __post_init__(self):
        if self.trajectories is None:
            self.trajectories = {}
        assert isinstance(self.trajectories,dict)
    
    @property
    def AC_num(self) -> int:
        return len(self.trajectories)
    
    @property
    def ids(self) -> typing.KeysView[int]:
        return self.trajectories.keys()
    
    def add_path(self,path:Path,stats:ACStats):
        if stats.id in self.trajectories:
            raise KeyError(f"AC id {stats.id} is already used")
        
        self.trajectories[stats.id] = (stats,path)

    @property
    def starts(self) -> dict[int,Pose3D]:
        output = dict()
        for s,traj in self.trajectories.values():
            output[s.id] = traj.start
        return output
    
    @property
    def ends(self) -> dict[int,Pose3D]:
        output = dict()
        for s,traj in self.trajectories.values():
            output[s.id] = traj.end
        return output
    
    def get_path(self,ac_id:int) -> tuple[ACStats,Path]:
        return self.trajectories[ac_id]
    
    def poses_at(self,t:float) -> dict[int,Pose3D]:
        output = dict()
        wind_dx = self.wind_x*t
        wind_dy = self.wind_y*t
        
        for s,traj in self.trajectories.values():
            p = traj.pose_at_time(t)
            p.x += wind_dx
            p.y += wind_dy
            output[s.id] = p
            
        return output


    def generate_list_of_poses(self,times:list[float]|np.ndarray) -> ListOfTimedPoses:
        output = []
        for t in times:
            poses = self.poses_at(t)
            output.append((t,poses))
        return output
    
    def sample_poses(self,sample_num:int) -> ListOfTimedPoses:
        times = np.linspace(0,self.duration,sample_num,endpoint=False)
        return self.generate_list_of_poses(times)
    
    def sample_at_fps(self,fps:int) -> ListOfTimedPoses:
        return self.sample_poses(np.ceil(self.duration*fps))
    
    def remove_path(self,ac_id:int|typing.Iterable[int]):
        if isinstance(ac_id,int):
            ids = [ac_id]
        else:
            ids = ac_id
        
        for id in ids:
            del self.trajectories[id]
    
    def generate_id_name_dict(self) -> dict[int,str]:
        output = dict()
        for s,p in self.trajectories.values():
            output[s.id] = p.abbr()
        return output
    
    def asdict(self,subset:typing.Optional[typing.Iterable[int]]=None) -> dict[str,typing.Any]:
        output = asdict(self)
        output.pop("trajectories")
        output["trajectories"] = []
        
        if subset is None:
            subset = self.trajectories.keys()

        ac_count = 0
        
        for id in subset:
            ac_count += 1
            t = self.trajectories[id]
            output["trajectories"].append(
                {
                    "stats" : t[0].asdict(),
                    "path"  : t[1].asdict()
                }
            )
            
        output["AC_num"] = ac_count

        return output
    
    def join(self,other:FleetPlan):
        
        # Check compatibility first
        assert(self.AC_num == other.AC_num)
        assert(self.z_alpha == other.z_alpha)
        assert(self.wind_x == other.wind_x)
        assert(self.wind_y == other.wind_y)
        assert(self.trajectories.keys() == other.trajectories.keys())
        
        for o_stats,o_traj in other.trajectories.values():
            s_stats,_ = self.trajectories[o_stats.id]
            assert(s_stats == o_stats)
        
        # Then perform manipulations
        self.separation = min(self.separation,other.separation)
        self.duration += other.duration
        
        for o_stats,o_traj in other.trajectories.values():
            _,s_traj = self.trajectories[o_stats.id]
            s_traj.join(o_traj)
            
    
    def merge(self,other:FleetPlan,override:bool=True):
        assert(self.separation == other.separation)
        assert(self.z_alpha == other.z_alpha)
        assert(self.wind_x == other.wind_x)
        assert(self.wind_y == other.wind_y)
        assert(self.duration >= 0)
        assert(other.duration >= 0)
        
        self.duration = max(self.duration,other.duration)
        
        for o_stats,o_traj in other.trajectories.values():
            try:
                if override:
                    self.trajectories[o_stats.id] = o_stats,o_traj
            except KeyError:
                self.trajectories[o_stats.id] = (o_stats,o_traj)
                        
    def follow_for(self,time:float) -> FleetPlan:
        new_trajs:dict[int,tuple[ACStats,Path]] = {}
        
        for stats,path in self.trajectories.values():
            
            new_path = path.follow_for(stats.airspeed*time)
            if new_path is None:
                continue
            else:
                new_path.shift(self.wind_x*time,self.wind_y*time)
                new_trajs[stats.id] = (stats,new_path)
        
        return FleetPlan(
            self.separation,
            self.z_alpha,
            self.wind_x,
            self.wind_y,
            self.duration-time,
            new_trajs
        )    
        
    