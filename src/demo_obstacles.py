#!/usr/bin/python3

import typing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from DubinsFleetPlanner.Poses import poses_XY_dist
from DubinsFleetPlanner.Dubins import BasicPath,DubinsMove,Pose2D
from DubinsFleetPlanner.DubinsPathFitting import plan_LSL,plan_RSL,plan_SLS,plan_SRS,ACStats
from DubinsFleetPlanner.UI.plotting import plot_BasicPath_obstacle


from scipy.optimize import direct,OptimizeResult

def done(ax:Axes):
    ax.set_aspect('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()
    exit(0)
    
def min_path_dist(p:BasicPath,pbis:BasicPath):
    def opt_fun(v):
        l1 = p.pose_at(v[0])
        l2 = pbis.pose_at(v[1])
        return poses_XY_dist(l1,l2)
    return direct(opt_fun,[(0,p.duration()),(0,pbis.duration())])

def min_straight_lstsq(p:BasicPath,pbis:BasicPath):
    assert p.type is DubinsMove.STRAIGHT
    assert pbis.type is DubinsMove.STRAIGHT
    
    s1 = np.array(p.start().astuple()[:2])
    s2 = np.array(pbis.start().astuple()[:2])
    
    b = s1 - s2
    A = np.array(
        [
            [-p.p1,pbis.p1],
            [-p.p2,pbis.p2]
        ]
    )
    print(A,b)
    
    return np.linalg.lstsq(A,b)

fig,ax = plt.subplots()

start_pose = Pose2D(0,1,-np.pi/8)
end_pose = Pose2D(1.5,-1,-np.pi*3/4)
stats = ACStats(0,1,1,1)

path = plan_SLS(stats,start_pose,end_pose)
assert path is not None

o = BasicPath(
    DubinsMove.STRAIGHT,
    20,
    6,
    -10,
    0,
    0,
    1,
    0,
    0
)

windx = 0.33
windy = 0

##### Plotting #####

first = True
travelled = 0

ax.quiver(start_pose.x, start_pose.y, np.cos(start_pose.angle), np.sin(start_pose.angle), angles='xy', color='g',label='Start Pose')
ax.quiver(end_pose.x, end_pose.y, np.cos(end_pose.angle), np.sin(end_pose.angle), angles='xy', color='r',label='End Pose')
ax.quiver(end_pose.x+path.total_length*windx, end_pose.y+path.total_length*windy, np.cos(end_pose.angle), np.sin(end_pose.angle), angles='xy', color='darkred',label='End Pose with wind')
ax.quiver(0,0.2,windx,windy,angles='xy',color='lightgray',label='Wind vector')

for p in path.sections:
    line = plot_BasicPath_obstacle(ax,p)[0]
    line.set_color('blue')
    if first:
        line.set_label('Path without wind')
    
    if p.type is not DubinsMove.STRAIGHT:
        _c = ax.scatter([p.x],[p.y],color='blue',marker='o',alpha=0.7)
        _start = p.start()
        _end = p.end()
        ax.plot([_start.x,p.x,_end.x],[_start.y,p.y,_end.y],color='blue',linestyle='dotted',alpha=0.7)
        if first:
            _c.set_label('Circle center')
            
        tpts = [(t,p.pose_at(t)) for t in np.linspace(0,p.duration(),100)]
        low_res_tpts = [(t,p.pose_at(t)) for t in np.linspace(0,p.duration(),5)]
        
        wind_pts = [Pose2D(pt.x+windx*(travelled+t),pt.y+windy*(travelled+t),pt.theta) for t,pt in tpts]
        
        low_res_wind_pts = [Pose2D(pt.x+windx*(travelled+t),pt.y+windy*(travelled+t),pt.theta) for t,pt in low_res_tpts]
        wind_center = [(p.x+windx*(travelled+t),p.y+windy*(travelled+t)) for t,_ in low_res_tpts]
        
        
        ax.plot([pt.x for pt in wind_pts],[pt.y for pt in wind_pts],color='purple',linestyle='solid',alpha=0.7)
        ax.plot([pt.x for pt in low_res_wind_pts],[pt.y for pt in low_res_wind_pts],color='purple',linestyle='dashdot',alpha=0.7,label='Approximated circular path with wind')
        ax.plot([wind_center[0][0],wind_center[-1][0]],[wind_center[0][1],wind_center[-1][1]],color='purple',marker='o',linestyle='dotted',alpha=0.7,label='Drifted circle center')
        ax.plot([wind_center[0][0],low_res_wind_pts[0].x],[wind_center[0][1],low_res_wind_pts[0].y],color='purple',linestyle='dotted',alpha=0.7)
        ax.plot([wind_center[-1][0],low_res_wind_pts[-1].x],[wind_center[-1][1],low_res_wind_pts[-1].y],color='purple',linestyle='dotted',alpha=0.7)
        
    
    
    else:
        _start = p.start()
        _end = p.end()
        _w_start = Pose2D(_start.x+windx*travelled,_start.y+windy*travelled,_start.theta)
        _w_end = Pose2D(_end.x+windx*(travelled+p.length),_end.y+windy*(travelled+p.length),_end.theta)
        if first:
            ax.plot([_w_start.x,_w_end.x],[_w_start.y,_w_end.y],color='purple',linestyle='solid',alpha=0.7,label='Path with wind')
        else:
            ax.plot([_w_start.x,_w_end.x],[_w_start.y,_w_end.y],color='purple',linestyle='solid',alpha=0.7)
        
        # ax.plot([_start.x,_w_start.x],[_start.y,_w_start.y],color='grey',linestyle='dashed',alpha=0.3)
        # ax.plot([_end.x,_w_end.x],[_end.y,_w_end.y],color='grey',linestyle='dashed',alpha=0.3)
        
        
        
    if first:
        first = False
    travelled += p.length
    
        
xlims = ax.get_xlim()
ylims = ax.get_ylim()

o_line = plot_BasicPath_obstacle(ax,o)[0]
o_line.set_color('black')
o_line.set_label('Obstacle')

ax.set_xlim(xlims)
ax.set_ylim(ylims)

ax.set_axis_off()

done(ax)