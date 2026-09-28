#!/usr/bin/python3

import typing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from DubinsFleetPlanner.Poses import poses_XY_dist
from DubinsFleetPlanner.Dubins import BasicPath,DubinsMove
from DubinsFleetPlanner.UI.plotting import plot_BasicPath_obstacle


from scipy.optimize import direct,OptimizeResult

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

o = BasicPath(
    DubinsMove.STRAIGHT,
    367.05985398922257,
    560570.02065107087,
    6264387.7554701474,
    0,
    -0.26811852988960672,
    -0.96338593197629585,
    0,0
)

obis = BasicPath(
    DubinsMove.STRAIGHT,
    565.62581897748487,
    560471.60510263778,
    6264034.1351706209,
    0,
    -0.97597002458732129,
    0.21790482121105831,
    0,
    0
)

obbis = BasicPath(
    DubinsMove.STRAIGHT,
    554.92984677851621,
    560019.22711944743,
    6264455.3840171332,
    0,
    0.9925462377288008,
    -0.12186864227684893,
    0,
    0
)

obbbis = BasicPath(
    DubinsMove.STRAIGHT,
    367.05985398922257,
    560570.02065107087,
    6264387.7554701474,
    0,
    -0.26811852988960672,
    -0.96338593197629585,
    0,
    0
)

plot_BasicPath_obstacle(ax,o)[0].set_label('Obstacle')

x=560291.23245170689
y=6264211.8156874133
z=0
v1=0.21275925779930938
v2=0.97710465059822904
v3=1.0558624882826717
v4=0
m=DubinsMove.STRAIGHT
length=94.408894350848684

p1 = BasicPath(
    m,
    length,
    x,y,z,
    v1,v2,v3,v4
)
p1_line = plot_BasicPath_obstacle(ax,p1)[0]
p1_line.set_color('b')
p1_line.set_label('Candidate')

# print(min_path_dist(p1,o))
# print(min_straight_lstsq(p1,o))

x_2=560350.40300402255
y_2=6264295.5526868291
z_2=0
v1_2=40
v2_2=-0.025000000000000001
v3_2=2.0122086898494311
v4_2=2.927194645492265
m_2=DubinsMove.RIGHT
length_2=179.91963889148647
p2 = BasicPath(
    m_2,
    length_2,
    x_2,y_2,z_2,
    v1_2,v2_2,v3_2,v4_2
)
plot_BasicPath_obstacle(ax,p2)[0].set_color('b')

x_3=560350.40300402255
y_3=6264255.5526868291
z_3=0
v1_3=-1
v2_3=0
v3_3=-67.777683856776548
v4_3=0
m_3=DubinsMove.STRAIGHT
length_3=92.542398951443545
p3 = BasicPath(
    m_3,
    length_3,
    x_3,y_3,z_3,
    v1_3,v2_3,v3_3,v4_3
)
plot_BasicPath_obstacle(ax,p3)[0].set_color('b')

startx = 560291.23245170689
starty = 6264211.8156874133
start_theta = 1.3563983186973685
endx = 560343.65128891508
endy = 6264255.5526868291
end_theta = 3.1415926535897931
ax.quiver(startx, starty, np.cos(start_theta), np.sin(start_theta), angles='xy', color='g',label='Start Pose')
ax.quiver(endx, endy, np.cos(end_theta), np.sin(end_theta), angles='xy', color='r',label='End Pose')
done(ax)


p4 = BasicPath(
    DubinsMove.STRAIGHT,
    1011.5162437970581,
    -11.51624379705811,
    -600,
    0,
    1,
    0,
    0,
    0
)
plot_BasicPath_obstacle(ax,p4)[0].set_color('b')

p5 = BasicPath(
    DubinsMove.STRAIGHT,
    505.75812189852905,
    494.24187810147095,
    -600,
    0,
    1,
    0,
    0,
    0
)
plot_BasicPath_obstacle(ax,p5)[0].set_color('b')