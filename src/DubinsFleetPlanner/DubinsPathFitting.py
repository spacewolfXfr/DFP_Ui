#!/usr/bin/python3

import typing

import numpy as np

from scipy.optimize import root_scalar

from .Dubins import LSL,RSR,LSR,RSL,RLR,LRL,SRS,SLS,all_paths,\
                    ACStats,Path,BasicPath,Pose2D,Pose3D,DubinsMove

#################### Basic planning ####################

def plan_LSL(stats:ACStats,start:Pose2D,end:Pose2D) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)
    
    sol = LSL(s,e)
    if sol is None:
        return None
    
    t,u,v = sol
    
    first = BasicPath.from_2D(DubinsMove.LEFT,t*r,start,r)
    
    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.STRAIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.LEFT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    if np.hypot(end.x - third_end.x,end.y - third_end.y) > 1e-9:
        return None

    return Path(r*(t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                [first,second,third])
    

def plan_RSR(stats:ACStats,start:Pose2D,end:Pose2D) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    sol = RSR(s,e)
    if sol is None:
        return None
    
    t,u,v = sol
    
    first = BasicPath.from_2D(DubinsMove.RIGHT,t*r,start,r)
    
    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.STRAIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.RIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    if np.hypot(end.x - third_end.x,end.y - third_end.y) > 1e-9:
        return None

    return Path(r*(t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                [first,second,third])
    

def plan_LSR(stats:ACStats,start:Pose2D,end:Pose2D) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    sol = LSR(s,e)
    if sol is None:
        return None
    
    t,u,v = sol

    first = BasicPath.from_2D(DubinsMove.LEFT,t*r,start,r)
    
    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.STRAIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.RIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    if np.hypot(end.x - third_end.x,end.y - third_end.y) > 1e-9:
        return None

    return Path(r*(t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                [first,second,third])
    

def plan_RSL(stats:ACStats,start:Pose2D,end:Pose2D) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    sol = RSL(s,e)
    if sol is None:
        return None
    
    t,u,v = sol

    first = BasicPath.from_2D(DubinsMove.RIGHT,t*r,start,r)
    
    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.STRAIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.LEFT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    if np.hypot(end.x - third_end.x,end.y - third_end.y) > 1e-9:
        return None

    return Path(r*(t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                [first,second,third])
    

def plan_RLR(stats:ACStats,start:Pose2D,end:Pose2D) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    sol = RLR(s,e)
    if sol is None:
        return None
    
    t,u,v = sol

    first = BasicPath.from_2D(DubinsMove.RIGHT,t*r,start,r)
    
    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.LEFT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.RIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    if np.hypot(end.x - third_end.x,end.y - third_end.y) > 1e-9:
        return None

    return Path(r*(t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                [first,second,third])
    

def plan_LRL(stats:ACStats,start:Pose2D,end:Pose2D) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    sol = LRL(s,e)
    if sol is None:
        return None
    
    t,u,v = sol

    first = BasicPath.from_2D(DubinsMove.LEFT,t*r,start,r)
    
    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.RIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.LEFT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    if np.hypot(end.x - third_end.x,end.y - third_end.y) > 1e-9:
        return None

    return Path(r*(t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                [first,second,third])
    

def plan_SRS(stats:ACStats,start:Pose2D,end:Pose2D) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    sol = SRS(s,e)
    if sol is None:
        return None
    
    t,u,v = sol

    first = BasicPath.from_2D(DubinsMove.STRAIGHT,t*r,start,r)
    
    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.RIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.STRAIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    if np.hypot(end.x - third_end.x,end.y - third_end.y) > 1e-9:
        return None

    return Path(r*(t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                [first,second,third])
    

def plan_SLS(stats:ACStats,start:Pose2D,end:Pose2D) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    sol = SLS(s,e)
    if sol is None:
        return None
    
    t,u,v = sol

    first = BasicPath.from_2D(DubinsMove.STRAIGHT,t*r,start,r)
    
    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.LEFT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.STRAIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    if np.hypot(end.x - third_end.x,end.y - third_end.y) > 1e-9:
        return None

    return Path(r*(t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                [first,second,third])

#################### Length fitting ####################

### Base length functions

def LRL_length(start:Pose2D, end:Pose2D) -> float:
    ans = LRL(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def RLR_length(start:Pose2D, end:Pose2D) -> float:
    ans = RLR(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def LSL_length(start:Pose2D, end:Pose2D) -> float:
    ans = LSL(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def LSR_length(start:Pose2D, end:Pose2D) -> float:
    ans = LSR(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)

def RSL_length(start:Pose2D, end:Pose2D) -> float:
    ans = RSL(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def RSR_length(start:Pose2D, end:Pose2D) -> float:
    ans = RSR(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def SLS_length(start:Pose2D, end:Pose2D) -> float:
    ans = SLS(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def SRS_length(start:Pose2D, end:Pose2D) -> float:
    ans = SRS(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    

### Generic fitting

def __fit_radius(length_fun:typing.Callable[[Pose2D,Pose2D],float],start:Pose2D,end:Pose2D, min_r:float,max_r:float, target:float) -> typing.Optional[float]:
    obj = lambda r : r*length_fun(Pose2D(start.x/r,start.y/r,start.angle),Pose2D(end.x/r,end.y/r,end.angle)) - target
    
    try:
        sol = root_scalar(obj,bracket=[min_r,max_r])
        if sol.converged:
            return sol.root
        else:
            return None
    except ValueError:
        return None
        
    
def __fit_endlengths(length_fun:typing.Callable[[Pose2D,Pose2D],float],start:Pose2D,end:Pose2D,ratio:float, target:float) -> typing.Optional[float]:
    assert 0 <= ratio <= 1
    
    min_l = 0.
    max_l = target
    
    vsx = np.cos(start.angle)
    vsy = np.sin(start.angle)
    
    vex = np.cos(end.angle)
    vey = np.sin(end.angle)
    
    
    obj = lambda l : length_fun(Pose2D(start.x+ratio*l*vsx,start.y+ratio*l*vsy,start.angle),
                                Pose2D(end.x-(1-ratio)*l*vex,end.y-(1-ratio)*l*vey,end.angle)) + l - target
    

    if np.isfinite(obj(max_l)):
        while not(np.isfinite(obj(min_l))):
            min_l += target/100 

    try:
        sol = root_scalar(obj,bracket=[min_l,max_l])
    except ValueError:
        return None
    
    
    if sol.converged:
        quality = obj(sol.root)
        if abs(quality) < 1e-6: 
            return sol.root
    else:
        return None
    
### Specific fitting

def fit_LRL_radius(start:Pose2D, end:Pose2D, r_min:float, r_max:float, target:float) -> typing.Optional[float]:
    return __fit_radius(LRL_length,start,end,r_min,r_max,target)

def fit_RLR_radius(start:Pose2D, end:Pose2D, r_min:float, r_max:float, target:float) -> typing.Optional[float]:
    return __fit_radius(RLR_length,start,end,r_min,r_max,target)

def fit_LSL_radius(start:Pose2D, end:Pose2D, r_min:float, r_max:float, target:float) -> typing.Optional[float]:
    return __fit_radius(LSL_length,start,end,r_min,r_max,target)

def fit_LSR_radius(start:Pose2D, end:Pose2D, r_min:float, r_max:float, target:float) -> typing.Optional[float]:
    return __fit_radius(LSR_length,start,end,r_min,r_max,target)

def fit_RSL_radius(start:Pose2D, end:Pose2D, r_min:float, r_max:float, target:float) -> typing.Optional[float]:
    return __fit_radius(RSL_length,start,end,r_min,r_max,target)

def fit_RSR_radius(start:Pose2D, end:Pose2D, r_min:float, r_max:float, target:float) -> typing.Optional[float]:
    return __fit_radius(RSR_length,start,end,r_min,r_max,target)

def fit_SLS_radius(start:Pose2D, end:Pose2D, r_min:float, r_max:float, target:float) -> typing.Optional[float]:
    return __fit_radius(SLS_length,start,end,r_min,r_max,target)

def fit_SRS_radius(start:Pose2D, end:Pose2D, r_min:float, r_max:float, target:float) -> typing.Optional[float]:
    return __fit_radius(SRS_length,start,end,r_min,r_max,target)



def fit_LRL_endpoints(start:Pose2D, end:Pose2D, ratio:float, target:float) -> typing.Optional[float]:
    return __fit_endlengths(LRL_length,start,end,ratio,target)

def fit_RLR_endpoints(start:Pose2D, end:Pose2D, ratio:float, target:float) -> typing.Optional[float]:
    return __fit_endlengths(RLR_length,start,end,ratio,target)

def fit_LSL_endpoints(start:Pose2D, end:Pose2D, ratio:float, target:float) -> typing.Optional[float]:
    return __fit_endlengths(LSL_length,start,end,ratio,target)

def fit_LSR_endpoints(start:Pose2D, end:Pose2D, ratio:float, target:float) -> typing.Optional[float]:
    return __fit_endlengths(LSR_length,start,end,ratio,target)

def fit_RSL_endpoints(start:Pose2D, end:Pose2D, ratio:float, target:float) -> typing.Optional[float]:
    return __fit_endlengths(RSL_length,start,end,ratio,target)

def fit_RSR_endpoints(start:Pose2D, end:Pose2D, ratio:float, target:float) -> typing.Optional[float]:
    return __fit_endlengths(RSR_length,start,end,ratio,target)

def fit_SLS_endpoints(start:Pose2D, end:Pose2D, ratio:float, target:float) -> typing.Optional[float]:
    return __fit_endlengths(SLS_length,start,end,ratio,target)

def fit_SRS_endpoints(start:Pose2D, end:Pose2D, ratio:float, target:float) -> typing.Optional[float]:
    return __fit_endlengths(SRS_length,start,end,ratio,target)


### Planning from fitting


def plan_LSL_from_straights(stats:ACStats,start:Pose2D,end:Pose2D, target:float, ratio:float) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)
    
    l = fit_LSL_endpoints(s,e,ratio,target/r)
    if l is None:
        return None
        
    sx = s.x + l*ratio*np.cos(s.angle)
    sy = s.y + l*ratio*np.sin(s.angle)

    ex = e.x - l*(1-ratio)*np.cos(e.angle)
    ey = e.y - l*(1-ratio)*np.sin(e.angle)

    sol = LSL(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle))
    if sol is None:
        return None
    
    t,u,v = sol
        
    
    sections: list[BasicPath] = []
    if ratio > 0:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*ratio,start,r))
        restart = sections[0].pose_at(l*r*ratio).to2D()
    else:
        restart = start

    first = BasicPath.from_2D(DubinsMove.LEFT,t*r,restart,r)
    
    sections.append(first)

    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.STRAIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    sections.append(second)

    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.LEFT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    sections.append(third)

    if ratio < 1:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*(1-ratio),Pose2D(third_end.x,third_end.y,third_end.theta),r))
        true_end = sections[-1].pose_at(l*r*(1-ratio))
    else:
        true_end = third_end

    if np.hypot(end.x - true_end.x,end.y - true_end.y) > 1e-9:
        return None

    return Path(r*(l+t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                sections)
    

def plan_RSR_from_straights(stats:ACStats,start:Pose2D,end:Pose2D, target:float, ratio:float) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    l = fit_RSR_endpoints(s,e,ratio,target/r)

    if l is None:
        return None
    
    sx = s.x + l*ratio*np.cos(s.angle)
    sy = s.y + l*ratio*np.sin(s.angle)

    ex = e.x - l*(1-ratio)*np.cos(e.angle)
    ey = e.y - l*(1-ratio)*np.sin(e.angle)

    sol = RSR(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle))
    if sol is None:
        return None
    
    t,u,v = sol

    
    sections: list[BasicPath] = []
    if ratio > 0:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*ratio,start,r))
        restart = sections[0].pose_at(l*r*ratio).to2D()
    else:
        restart = start

    first = BasicPath.from_2D(DubinsMove.RIGHT,t*r,restart,r)
    
    sections.append(first)

    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.STRAIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    sections.append(second)

    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.RIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    sections.append(third)

    if ratio < 1:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*(1-ratio),Pose2D(third_end.x,third_end.y,third_end.theta),r))
        true_end = sections[-1].pose_at(l*r*(1-ratio))
    else:
        true_end = third_end

    if np.hypot(end.x - true_end.x,end.y - true_end.y) > 1e-9:
        return None

    return Path(r*(l+t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                sections)
    

def plan_LSR_from_straights(stats:ACStats,start:Pose2D,end:Pose2D, target:float, ratio:float) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    l = fit_LSR_endpoints(s,e,ratio,target/r)
    if l is None:
        return None
    
    sx = s.x + l*ratio*np.cos(s.angle)
    sy = s.y + l*ratio*np.sin(s.angle)

    ex = e.x - l*(1-ratio)*np.cos(e.angle)
    ey = e.y - l*(1-ratio)*np.sin(e.angle)

    sol = LSR(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle))
    if sol is None:
        return None
    
    t,u,v = sol

    sections: list[BasicPath] = []
    if ratio > 0:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*ratio,start,r))
        restart = sections[0].pose_at(l*r*ratio).to2D()
    else:
        restart = start

    first = BasicPath.from_2D(DubinsMove.LEFT,t*r,restart,r)
    
    sections.append(first)

    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.STRAIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    sections.append(second)

    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.RIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    sections.append(third)

    if ratio < 1:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*(1-ratio),Pose2D(third_end.x,third_end.y,third_end.theta),r))
        true_end = sections[-1].pose_at(l*r*(1-ratio))
    else:
        true_end = third_end

    if np.hypot(end.x - true_end.x,end.y - true_end.y) > 1e-9:
        return None

    return Path(r*(l+t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                sections)
    

def plan_RSL_from_straights(stats:ACStats,start:Pose2D,end:Pose2D, target:float, ratio:float) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    l = fit_RSL_endpoints(s,e,ratio,target/r)
    if l is None:
        return None
    
    sx = s.x + l*ratio*np.cos(s.angle)
    sy = s.y + l*ratio*np.sin(s.angle)

    ex = e.x - l*(1-ratio)*np.cos(e.angle)
    ey = e.y - l*(1-ratio)*np.sin(e.angle)

    sol = RSL(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle))
    if sol is None:
        return None
    
    t,u,v = sol

    sections: list[BasicPath] = []
    if ratio > 0:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*ratio,start,r))
        restart = sections[0].pose_at(l*r*ratio).to2D()
    else:
        restart = start

    first = BasicPath.from_2D(DubinsMove.RIGHT,t*r,restart,r)
    
    sections.append(first)

    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.STRAIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    sections.append(second)

    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.LEFT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    sections.append(third)

    if ratio < 1:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*(1-ratio),Pose2D(third_end.x,third_end.y,third_end.theta),r))
        true_end = sections[-1].pose_at(l*r*(1-ratio))
    else:
        true_end = third_end

    if np.hypot(end.x - true_end.x,end.y - true_end.y) > 1e-9:
        return None

    return Path(r*(l+t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                sections)
    

def plan_RLR_from_straights(stats:ACStats,start:Pose2D,end:Pose2D, target:float, ratio:float) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    l = fit_RLR_endpoints(s,e,ratio,target/r)
    if l is None:
        return None
    
    sx = s.x + l*ratio*np.cos(s.angle)
    sy = s.y + l*ratio*np.sin(s.angle)

    ex = e.x - l*(1-ratio)*np.cos(e.angle)
    ey = e.y - l*(1-ratio)*np.sin(e.angle)

    sol = RLR(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle))
    if sol is None:
        return None
    
    t,u,v = sol

    sections: list[BasicPath] = []
    if ratio > 0:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*ratio,start,r))
        restart = sections[0].pose_at(l*r*ratio).to2D()
    else:
        restart = start

    first = BasicPath.from_2D(DubinsMove.RIGHT,t*r,restart,r)
    
    sections.append(first)

    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.LEFT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    sections.append(second)

    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.RIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    sections.append(third)

    if ratio < 1:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*(1-ratio),Pose2D(third_end.x,third_end.y,third_end.theta),r))
        true_end = sections[-1].pose_at(l*r*(1-ratio))
    else:
        true_end = third_end

    if np.hypot(end.x - true_end.x,end.y - true_end.y) > 1e-9:
        return None

    return Path(r*(l+t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                sections)
    

def plan_LRL_from_straights(stats:ACStats,start:Pose2D,end:Pose2D, target:float, ratio:float) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    l = fit_LRL_endpoints(s,e,ratio,target/r)
    if l is None:
        return None
    
    sx = s.x + l*ratio*np.cos(s.angle)
    sy = s.y + l*ratio*np.sin(s.angle)

    ex = e.x - l*(1-ratio)*np.cos(e.angle)
    ey = e.y - l*(1-ratio)*np.sin(e.angle)

    sol = LRL(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle))
    if sol is None:
        return None
    
    t,u,v = sol

    sections: list[BasicPath] = []
    if ratio > 0:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*ratio,start,r))
        restart = sections[0].pose_at(l*r*ratio).to2D()
    else:
        restart = start

    first = BasicPath.from_2D(DubinsMove.LEFT,t*r,restart,r)
    
    sections.append(first)

    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.RIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    sections.append(second)

    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.LEFT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    sections.append(third)

    if ratio < 1:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*(1-ratio),Pose2D(third_end.x,third_end.y,third_end.theta),r))
        true_end = sections[-1].pose_at(l*r*(1-ratio))
    else:
        true_end = third_end

    if np.hypot(end.x - true_end.x,end.y - true_end.y) > 1e-9:
        return None

    return Path(r*(l+t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                sections)
    

def plan_SRS_from_straights(stats:ACStats,start:Pose2D,end:Pose2D, target:float, ratio:float) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    l = fit_SRS_endpoints(s,e,ratio,target/r)
    if l is None:
        return None
    
    sx = s.x + l*ratio*np.cos(s.angle)
    sy = s.y + l*ratio*np.sin(s.angle)

    ex = e.x - l*(1-ratio)*np.cos(e.angle)
    ey = e.y - l*(1-ratio)*np.sin(e.angle)

    sol = SRS(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle))
    if sol is None:
        return None
    
    t,u,v = sol

    sections: list[BasicPath] = []
    if ratio > 0:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*ratio,start,r))
        restart = sections[0].pose_at(l*r*ratio).to2D()
    else:
        restart = start

    first = BasicPath.from_2D(DubinsMove.STRAIGHT,t*r,restart,r)
    
    sections.append(first)

    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.RIGHT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    sections.append(second)

    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.STRAIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    sections.append(third)

    if ratio < 1:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*(1-ratio),Pose2D(third_end.x,third_end.y,third_end.theta),r))
        true_end = sections[-1].pose_at(l*r*(1-ratio))
    else:
        true_end = third_end

    if np.hypot(end.x - true_end.x,end.y - true_end.y) > 1e-9:
        return None

    return Path(r*(l+t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                sections)
    

def plan_SLS_from_straights(stats:ACStats,start:Pose2D,end:Pose2D, target:float, ratio:float) -> typing.Optional[Path]:
    r = stats.turn_radius
    s = Pose2D(start.x/r,start.y/r,start.angle)
    e = Pose2D(end.x/r,end.y/r,end.angle)

    l = fit_SLS_endpoints(s,e,ratio,target/r)
    if l is None:
        return None
    
    sx = s.x + l*ratio*np.cos(s.angle)
    sy = s.y + l*ratio*np.sin(s.angle)

    ex = e.x - l*(1-ratio)*np.cos(e.angle)
    ey = e.y - l*(1-ratio)*np.sin(e.angle)

    sol = SLS(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle))
    if sol is None:
        return None
    
    t,u,v = sol

    sections: list[BasicPath] = []
    if ratio > 0:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*ratio,start,r))
        restart = sections[0].pose_at(l*r*ratio).to2D()
    else:
        restart = start

    first = BasicPath.from_2D(DubinsMove.STRAIGHT,t*r,restart,r)
    
    sections.append(first)

    first_end = first.pose_at(t*r)
    second = BasicPath.from_2D(DubinsMove.LEFT, u*r, Pose2D(first_end.x,first_end.y,first_end.theta),r)
    
    sections.append(second)

    second_end = second.pose_at(u*r)
    third = BasicPath.from_2D(DubinsMove.STRAIGHT, v*r, Pose2D(second_end.x,second_end.y,second_end.theta),r)
    third_end = third.pose_at(v*r)

    sections.append(third)

    if ratio < 1:
        sections.append(BasicPath.from_2D(DubinsMove.STRAIGHT,l*r*(1-ratio),Pose2D(third_end.x,third_end.y,third_end.theta),r))
        true_end = sections[-1].pose_at(l*r*(1-ratio))
    else:
        true_end = third_end

    if np.hypot(end.x - true_end.x,end.y - true_end.y) > 1e-9:
        return None

    return Path(r*(l+t+u+v),
                Pose3D(start.x,start.y,0.,start.angle),
                Pose3D(end.x,end.y,0.,end.angle),
                sections)