from typing import Callable

import numpy as np
from scipy.interpolate import make_interp_spline,BSpline

from .Dubins import Path,Pose3D,DubinsMove,ACStats,BasicPath

def unit_interpolate_basicpath(p:BasicPath,stepping:float=1.) -> BSpline:
    start = p.start()
    end = p.end()
    
    vstart  = np.asarray([np.cos(start.theta),np.sin(start.theta)])
    vend    = np.asarray([np.cos(end.theta),np.sin(end.theta)])
        
    astart  = np.zeros(2)
    aend    = np.zeros(2)
    
    if p.type == DubinsMove.STRAIGHT:
        lengths = [0,p.length]
        ps = np.asarray([[start.x,start.y],[end.x,end.y]])
        
        return make_interp_spline(lengths,ps,k=5,bc_type=(
            [(1,vstart),(2,astart)],
            [(1,vend),(2,aend)]
        ))
    else:
        sample_num = int(np.ceil(p.length/stepping))
        lengths = np.linspace(0,p.length,sample_num,endpoint=True)
        
        poses = [p.pose_at(t) for t in lengths]
        ys = [[e.x,e.y] for e in poses]
        
        return make_interp_spline(lengths,ys,k=5,bc_type=(
            [(1,vstart),(2,astart)],
            [(1,vend),(2,aend)]
        ))
        
        
        

def interpolate_path(stats:ACStats,p:Path,cicle_step:float=1,junctions_offset:float=0.5) -> BSpline:
    
    ## Sample path
    sample_times = []
    sample_poses = []
    
    curr_time = 0
    
    for i,s in enumerate(p.sections):
        if i == 0:
            sample_times.append(curr_time)
            sample_poses.append(s.start())
        
        l = s.length - 2*junctions_offset
        if l > 0:
            n_samples = int(l/cicle_step)
            samples = np.linspace(junctions_offset, s.length-junctions_offset, n_samples,endpoint=True)
            for t in samples:
                sample_times.append(curr_time+t)
                sample_poses.append(s.pose_at(t))
        
        if i == len(p.sections)-1:
            sample_times.append(curr_time+s.length)
            sample_poses.append(s.end())
            
        curr_time += s.length
        
    ## Convert to numpy arrays
    sample_times = np.array(sample_times)
    sample_xy = np.array([[p.x,p.y] for p in sample_poses])
        
    ## Interpolate
    spline = make_interp_spline(
        sample_times,
        sample_xy,
        k=5,
        bc_type=([(1, [stats.airspeed*np.cos(p.start.theta),stats.airspeed*np.sin(p.start.theta)]), (2, [0,0])],
                 [(1, [stats.airspeed*np.cos(p.end.theta),stats.airspeed*np.sin(p.end.theta)]), (2, [0,0])],
    ))
    
    return spline

    
    