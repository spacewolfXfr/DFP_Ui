#!/usr/bin/python3

import typing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from DubinsFleetPlanner.Poses import poses_XY_dist
from DubinsFleetPlanner.Dubins import BasicPath,DubinsMove,ACStats,Path
from DubinsFleetPlanner.UI.plotting import plot_BasicPath_obstacle,plot_several_pose2d_sequences,plot_pose2d_sequence
from DubinsFleetPlanner.UI._plotting_extra import my_cmap
from DubinsFleetPlanner.figure import plot_separation_sections


from scipy.optimize import direct

STRAIGHT = DubinsMove.STRAIGHT
LEFT     = DubinsMove.LEFT
RIGHT    = DubinsMove.RIGHT

def done(ax:Axes):
    ax.set_aspect('equal')
    ax.legend()
    plt.show()
    exit(0)
    
def min_path_dist(p:BasicPath,pbis:BasicPath):
    def opt_fun(v):
        l1 = p.pose_at(v[0])
        l2 = pbis.pose_at(v[1])
        return poses_XY_dist(l1,l2)
    return direct(opt_fun,[(0,p.length),(0,pbis.length)])

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

s1 = BasicPath(**{
    'x':381.84549785164512,
    'y':3081.1155665326842,
    'z':0,
    'p1':-3.3332870018685861,
    'p2':-0.017574819630541628,
    'p3':0,
    'p4':0,
    'type':STRAIGHT,
    'length':3.1765989416110858,
})

s2 = BasicPath(**{
    'x':471.26012298871711,
    'y':3068.5155471287085,
    'z':0,
    'p1':1.0610329539459691,
    'p2':-3.1415926535897927,
    'p3':0,
    'p4':1.4283501130324778,
    'type':RIGHT,
    'length':3.9219964720370584,
})


plot_BasicPath_obstacle(ax,s1)
plot_BasicPath_obstacle(ax,s2)

# path1 = Path.from_sections_only([s1])
# path2 = Path.from_sections_only([s2])

# stat1 = ACStats(0x100,10/3,1,1)
# stat2 = ACStats(0x5f ,10/3,1,1)

# plot_separation_sections(path1,path2,stat1,stat2,ax,my_cmap,'all')
# done(ax)


p0 = BasicPath(**{
    'x':392.25945288564856,
    'y':3088.1081614421164,
    'z':0,
    'p1':-0.91401214653723639,
    'p2':-0.40568682007478812,
    'p3':0,
    'p4':0,
    'type':STRAIGHT,
    'length':10.000000000000011,
    })

p1 = BasicPath(**{
    'x':382.01836963645502,
    'y':3086.53175943719,
    'z':0,
    'p1':2.7138219171582056,
    'p2':-0.3684840164630831,
    'p3':0,
    'p4':-1.1530661870772634,
    'type':RIGHT,
    'length':0.1010618130002877,
})

p2 = BasicPath(**{
    'x':383.02621772968786,
    'y':3084.0120230186308,
    'z':0,
    'p1':-0.92848259593910387,
    'p2':-0.37137591337912979,
    'p3':0,
    'p4':0,
    'type':STRAIGHT,
    'length':8.4016494246751847,
})

p3 = BasicPath(**{
    'x':376.23328055492806,
    'y':3078.3721163710911,
    'z':0,
    'p1':2.7138219171582056,
    'p2':0.3684840164630831,
    'p3':0,
    'p4':1.9512868037471431,
    'type':LEFT,
    'length':16.033171146072313,
})

p4 = BasicPath(**{
    'x':376.21897207571675,
    'y':3081.0859005676461,
    'z':0,
    'p1':-0.99998610056057513,
    'p2':-0.0052724458892498731,
    'p3':0,
    'p4':0,
    'type':STRAIGHT,
    'length':9.9999999999999858,
})

p5 = BasicPath(**{
    'x':376.21897207571675,
    'y':3081.0859005676461,
    'z':0,
    'p1':-0.99998610056057513,
    'p2':-0.0052724458892498731,
    'p3':0,
    'p4':0,
    'type':STRAIGHT,
    'length':9.9999999999999858,
})

p = Path.from_sections_only([p0,p1,p2,p3,p4])

o0 = BasicPath(**{
    'x':335.62968800257232,
    'y':3089.0392216674772,
    'z':0,
    'p1':3.2995722718657579,
    'p2':-0.47321658238670333,
    'p3':0,
    'p4':0,
    'type':STRAIGHT,
    'length':41.151110906216516,
})

o1 = BasicPath(**{
    'x':376.21337783688034,
    'y':3082.1469187738285,
    'z':0,
    'p1':1.0610329539459691,
    'p2':-3.1415926535897927,
    'p3':0,
    'p4':7.7115354202120656,
    'type':RIGHT,
    'length':3.1765989416111147,
})

o2 = BasicPath(**{
    'x':376.21897207571675,
    'y':3081.0859005676461,
    'z':0,
    'p1':-3.3332870018685843,
    'p2':-0.017574819630832912,
    'p3':0,
    'p4':0,
    'type':STRAIGHT,
    'length':9.9999999999999858,
})


o = Path.from_sections_only([o0,o1,o2])

stat1 = ACStats(0x100,10/3,1,1)
stat2 = ACStats(0x5f ,10/3,1,1)

plot_separation_sections(p,o,stat1,stat2,ax,my_cmap,'all')

# startx = 560291.23245170689
# starty = 6264211.8156874133
# start_theta = 1.3563983186973685
# endx = 560343.65128891508
# endy = 6264255.5526868291
# end_theta = 3.1415926535897931
# ax.quiver(startx, starty, np.cos(start_theta), np.sin(start_theta), angles='xy', color='g',label='Start Pose')
# ax.quiver(endx, endy, np.cos(end_theta), np.sin(end_theta), angles='xy', color='r',label='End Pose')
done(ax)