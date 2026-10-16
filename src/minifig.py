#!/usr/bin/python3

import typing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from DubinsFleetPlanner.Poses import poses_XY_dist
from DubinsFleetPlanner.Dubins import BasicPath,DubinsMove,ACStats,Path,Pose3D,SRS,poses_euclidean_dist
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

# start = Pose3D(-1.6229542339982297,
#                 -1.0563755753005593,
#                 0,
#                 -2.4675849223753814).to2D()

# end = Pose3D(-4.175511861897613,
#                 -2.0659710592402876,
#                 0,
#                 0.37720150203297687).to2D()

# path_params = SRS(start,end)

# if path_params is None:
#     print("No path found")
#     exit(1)

# s1 = BasicPath.from_2D(STRAIGHT,path_params[0],start,1.)
# s2 = BasicPath.from_2D(RIGHT,path_params[1],s1.end().to2D(),1.)
# s3 = BasicPath.from_2D(STRAIGHT,path_params[2],s2.end().to2D(),1.)

# print("Lengths: ", path_params)
# print("Measured lengths: ", s1.length, s2.length, s3.length)
# print("Junctions: ", s1.end().to2D(), s2.end().to2D())
# print("Endpoints: ", start,end)

# p = Path.from_sections_only([s1,s2,s3])

# plot_pose2d_sequence(ax,p.sample(200), color='b',label='Path')

# done(ax)


s1 = BasicPath(**{
    'x':-2.369244240568174,
    'y':4.1730026404643965,
    'z':-0.6208162757504212,
    'p1':0.76730173826951442,
    'p2':0.87937680126526996,
    'p3':0.21097824506661703,
    'p4':-2.707146700567955,
    'type':LEFT,
    'length':100000000000000000
})

s2 = BasicPath(**{
    'x':-1.0538663585239547,
    'y':-1.2404960439600199,
    'z':1.568775591030982,
    'p1':3.1509526094997478,
    'p2':-0.95593980475171336,
    'p3':0.06646736759992547,
    'p4':-2.4899237240570358,
    'type':RIGHT,
    'length':100000000000000000,
})

stats1 = ACStats(**{
    "id":0x2,
    "airspeed":2.7044482018375078,
    "target_airspeed":2.7044482018375078,
    "turn_radius":2.4706326759123103,
    "climb":0.36415213914655742,
    "min_acceleration":-1,
    "max_acceleration":1,
})

stats2 = ACStats(**{
    "id":0x3,
    "airspeed":4.0066723050612847,
    "target_airspeed":4.0066723050612847,
    "turn_radius":3.4559490159639417,
    "climb":0.046228208719836038,
    "min_acceleration":-1,
    "max_acceleration":1,
})

p1 = Path.from_sections_only([s1])
p2 = Path.from_sections_only([s2])

times = np.linspace(0,10,1000)

poses1 = [t for t in zip(times,p1.poses_at_times(times,stats1))]
poses2 = [t for t in zip(times,p2.poses_at_times(times,stats2))]

print(len(poses1),len(times),len(poses2))

poses_dict:dict[int,list[tuple[float,Pose3D]]] = {
    0x2: poses1,
    0x3: poses2
}


dists = [poses_XY_dist(p1[1],p2[1]) for p1,p2 in zip(poses1,poses2)]
dtimes = times[:len(dists)]

mId = np.argmin(dists)
mTime = dtimes[mId]
mDist = dists[mId]


print(f"Minimum distance: {mDist} at time: {mTime}")
print(f"Trajectories durations: {times[len(poses1)-1]}, {times[len(poses2)-1]}")

plot_several_pose2d_sequences(ax,poses_dict,endpoints=True,label=True)

done(ax)

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