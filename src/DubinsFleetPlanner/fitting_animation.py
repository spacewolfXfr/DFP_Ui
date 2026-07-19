#!/usr/bin/python3

import typing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.axes import Axes
from matplotlib.lines import Line2D


from .Dubins import Pose2D,Pose3D,ACStats,Path,poses_XY_dist
from .plotting import plot_pose2d_sequence

from .DubinsPathFitting import plan_LSL, plan_RSR,\
        fit_LSL_radius,fit_RSR_radius,plan_LSL_from_straights,plan_RSR_from_straights

from .UI._plotting_extra import my_cmap6,my_cmap

def set_linedata_from_path(l:Line2D,path:typing.Optional[Path],num:int=200):
    if path is None:
        l.set_xdata([])
        l.set_ydata([])
    else:
        samples = path.sample(num)
        xs = []
        ys = []
        for p in samples:
            xs.append(p.x)
            ys.append(p.y)
        l.set_xdata(xs)
        l.set_ydata(ys)

if __name__ == '__main__':
    stats = ACStats(0,1.,0.,0.5)
    
    d = 2
    start_degrees = -20
    end_degrees = 330
    
    start   = Pose2D(0,0,np.deg2rad(start_degrees))
    end     = Pose2D(0,-d,np.deg2rad(end_degrees))
    
    pts_dist = poses_XY_dist(start,end)
    
    ## Plotting length variations
    
    length_multiplier = np.linspace(1.,3,1200)
    
    base_length_LSL = plan_LSL(stats,start,end)
    base_length_RSR = plan_RSR(stats,start,end)
    
    if base_length_LSL is None and base_length_RSR is None:
        print(f"Could not plan either LSL or RSR from {start} to {end} with stats {stats}")
        exit(1)
    
    if base_length_LSL is not None and base_length_RSR is not None:
        base_length = max(base_length_LSL.total_length,base_length_RSR.total_length)
    elif base_length_RSR is not None:
        base_length = base_length_RSR
    else:
        base_length = base_length_LSL
    
    titles      = ["Shift start", "Shift end", "Shift both", "Turn radius"]
    markers     = [5,4,7,6]
    colors      = [my_cmap6[0],my_cmap6[2],my_cmap6[4], my_cmap6[5]]
    
    fig,ax = plt.subplots(figsize=(16,9))
    
    ax.quiver(
        [start.x,end.x],
        [start.y,end.y],
        [np.cos(start.angle),np.cos(end.angle)],
        [np.sin(start.angle),np.sin(end.angle)],
        angles='xy',
        color='black',
        label='Endpoints'
    )
    
    LSL_radii = [fit_LSL_radius(start,end,stats.turn_radius,1e3*pts_dist,m*base_length/stats.airspeed) for m in length_multiplier]
    LSL_radius_plans = [
        None if r is None else 
            plan_LSL(ACStats(stats.id,stats.airspeed,stats.climb,r),start,end)
        for r in LSL_radii]
    
    RSR_radii = [fit_RSR_radius(start,end,stats.turn_radius,1e3*pts_dist,m*base_length/stats.airspeed) for m in length_multiplier]
    RSR_radius_plans = [
        None if r is None else 
            plan_RSR(ACStats(stats.id,stats.airspeed,stats.climb,r),start,end)
        for r in RSR_radii]
    
    LSL_straight_plans = [
        plan_LSL_from_straights(stats,start,end,m*base_length/stats.airspeed,0.5) for m in length_multiplier
    ]
    
    RSR_straight_plans = [
        plan_RSR_from_straights(stats,start,end,m*base_length/stats.airspeed,0.5) for m in length_multiplier
    ]
    
    maxpts:list[Pose3D] = []
    
    if last_valid := list(filter(lambda o : o is not None,LSL_radius_plans))[-1]:
        maxpts.extend(last_valid.sample(200))
    if last_valid := list(filter(lambda o : o is not None,RSR_radius_plans))[-1]:
        maxpts.extend(last_valid.sample(200))
    if last_valid := list(filter(lambda o : o is not None,LSL_straight_plans))[-1]:
        maxpts.extend(last_valid.sample(200))
    if last_valid := list(filter(lambda o : o is not None,RSR_straight_plans))[-1]:
        maxpts.extend(last_valid.sample(200))
        
    max_x = start.x
    min_x = start.x
    max_y = start.y
    min_y = start.y
    
    for p in maxpts:
        max_x = max(max_x,p.x)
        min_x = min(min_x,p.x)
        max_y = max(max_y,p.y)
        min_y = min(min_y,p.y)
    
    ax.set_xlim(min_x,max_x)
    ax.set_ylim(min_y,max_y)
    
    lsl_rad_line = ax.plot([],[],color='r',linestyle='-',label='LSL radius based')[0]
    lsl_straight_line = ax.plot([],[],color='r',linestyle=':',label='LSL straight based')[0]
    rsr_rad_line = ax.plot([],[],color='b',linestyle='-',label='RSR radius based')[0]
    rsr_straight_line = ax.plot([],[],color='b',linestyle=':',label='RSR straight based')[0]
    
    def update(frame:int):
        lsl_rad_plan = LSL_radius_plans[frame]
        lsl_straight_plan = LSL_straight_plans[frame]
        rsr_rad_plan = RSR_radius_plans[frame]
        rsr_straight_plan = RSR_straight_plans[frame]
        
        set_linedata_from_path(lsl_rad_line,lsl_rad_plan,100)
        set_linedata_from_path(lsl_straight_line,lsl_straight_plan,100)
        set_linedata_from_path(rsr_rad_line,rsr_rad_plan,100)
        set_linedata_from_path(rsr_straight_line,rsr_straight_plan,100)
        
        
        
        return [lsl_rad_line,lsl_straight_line,rsr_rad_line,rsr_straight_line,ax.set_title(f"Target length: {length_multiplier[frame]*base_length:.3f}",y=-0.01)]
            
    ax.set_axis_off()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.legend(loc='lower left')
    fig.tight_layout()
    
    ani = FuncAnimation(fig,update,len(length_multiplier),interval=1000//30,repeat=True,blit=False)
    ani.save("length_fitting.mp4")
    plt.show()
    exit(0)
