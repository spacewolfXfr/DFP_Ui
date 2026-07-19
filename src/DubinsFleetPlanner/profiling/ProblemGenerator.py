#!/usr/bin/python3

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

import typing,dataclasses,pathlib,copy

import csv

import numpy as np
from scipy.spatial.distance import pdist

from .ioUtils import AC_PP_Problem,write_pathplanning_problem_to_CSV

from .Dubins import Pose3D,ACStats

from .Formation import Formation

######################################## Case creator ########################################

class _RepulsionSimulator:
    
    def __init__(self,
                 pts:np.ndarray,
                 charges:typing.Optional[np.ndarray]=None,
                 speeds:typing.Optional[np.ndarray]=None,
                 force_factor:float=1.,
                 force_power:int = 2,
                 friction:float=1.):
        
        self.pts    = pts
        self.charges = charges if charges is not None else np.ones(len(pts),float)
        self.speeds = speeds if speeds is not None else np.zeros(pts.shape)
        
        assert len(self.pts) == len(self.charges)
        assert self.pts.shape == self.speeds.shape
        
        self.force_factor   = force_factor
        self.force_power    = force_power
        self.friction       = friction
        
    def _compute_forces(self) -> np.ndarray:
        # Compute diff vectors
        delta_matrix = self.pts[:,np.newaxis] - self.pts
        
        # Compute charges outer product
        charges_matrix = self.charges[:,np.newaxis] * self.charges
        
        # Compute distances
        dist_matrix = np.linalg.norm(delta_matrix,axis=-1)
        
        # Fill diagonal to avoid singularities
        np.fill_diagonal(dist_matrix,1.)
        
        # Use previous results for generalized Coulomb law
        forces_matrix = - delta_matrix*(self.force_factor*charges_matrix/np.pow(dist_matrix,self.force_power+1))[:,:,np.newaxis]
        
        # Sum on column
        summed_forces = np.sum(forces_matrix,axis=0)
        
        # Saturate to avoid weird shenanigans
        return np.clip(summed_forces,a_min=-np.inf,a_max=1e6)
    
    
    
    def run_until_time(self,t_end:float,dt:float=1e-3) -> tuple[np.ndarray,int]:
        t = 0.
        i = 0
        while t < t_end:
            forces = self._compute_forces()
            self.pts += self.speeds*dt
            self.speeds += (forces-np.abs(self.speeds)*self.speeds*self.friction)*dt
            t += dt
            i += 1
        
        return self.pts,i
        
    def run_until_distance(self,mdist:float,dt:float=1e-3) -> tuple[np.ndarray,int]:
        t = 0.
        i = 0
        mdist2 = mdist*mdist
        curr_mdist2 = np.min(pdist(self.pts,'sqeuclidean'))
        
        while curr_mdist2 < mdist2:
            forces = self._compute_forces()
            self.pts += self.speeds*dt
            
            # Multiplies the repulsion force by mdist² such that it is unit when mdist is achieved
            self.speeds += (mdist2*forces-np.abs(self.speeds)*self.speeds*self.friction)*dt
            t += dt
            i += 1
            curr_mdist2 = np.min(pdist(self.pts,'sqeuclidean'))
            
        return self.pts,i
    
    

class RandomPathPlanningGenerator:
    def __init__(self,seed:int=0,verbose:int=0) -> None:
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose
        
    def _sample_in_ball(self,min_r:float,max_r:float,max_angle:float=2*np.pi,
                        latitude_range:typing.Optional[tuple[float,float]]=None,
                        num:typing.Union[int,tuple[int,...]]=1) -> typing.Union[tuple[float,float,float] , tuple[np.ndarray,np.ndarray,np.ndarray]]:
        
        r = self.rng.uniform(min_r,max_r,num)
        a = self.rng.uniform(0,max_angle,num)
        phi = self.rng.uniform(latitude_range[0],latitude_range[1],num) if latitude_range is not None else np.zeros(num)
        
        return r*np.cos(a)*np.cos(phi),r*np.sin(a)*np.cos(phi),r*np.sin(phi)
    
    def _repulse_separate(self,pts:np.ndarray,mdist:float,dt:float=1e-3,**kwargs) -> np.ndarray:
        sol,iters = _RepulsionSimulator(pts,**kwargs).run_until_distance(mdist,dt)
        if self.verbose > 0:
            print(f"Iteration count: {iters}")
        
        return sol
        

    def _generate_uniform_pts(self,N:int,xrange:tuple[float,float],yrange:tuple[float,float],zrange:typing.Optional[tuple[float,float]]=None) -> np.ndarray:
        xs = self.rng.uniform(xrange[0],xrange[1],N)
        ys = self.rng.uniform(yrange[0],yrange[1],N)
        
        if zrange is not None:
            zs = self.rng.uniform(zrange[0],zrange[1],N)
        else:
            zs = np.zeros(N)
            
        pts = np.stack([xs,ys,zs]).T
            
        return pts
    
    def _generate_endpoints_in_disk(self,init_pts:np.ndarray,d_range:tuple[float,float],latitude_range:typing.Optional[tuple[float,float]]=None) -> np.ndarray:
        offsets = self._sample_in_ball(d_range[0],d_range[1],latitude_range=latitude_range,num=len(init_pts))
        offset_pts = np.stack(offsets).T
        
        return init_pts+offset_pts
    
    def _generate_uniform_separated_poses(self, mdist:float, N:int,
        xrange:tuple[float,float], yrange:tuple[float,float],
        zrange:typing.Optional[tuple[float,float]]=None,
        **repulse_kwargs) -> list[Pose3D]:
        
        og_pts = self._generate_uniform_pts(N,xrange,yrange,zrange)
        
        if self.verbose > 0:
            print(f"Separating {N} points: ",end='')
            
        sep_pts = self._repulse_separate(og_pts,mdist,**repulse_kwargs)
        angles = self.rng.uniform(0,2*np.pi,N)
        
        return [Pose3D(p[0],p[1],p[2],a) for p,a in zip(sep_pts,angles)]
    
    def generate_uniform_case_with_repulsion(
        self, mdist:float, N:int,
        xrange:tuple[float,float], yrange:tuple[float,float],
        zrange:typing.Optional[tuple[float,float]]=None,
        **repulse_kwargs) -> tuple[list[Pose3D],list[Pose3D]]:
        
        return self._generate_uniform_separated_poses(mdist,N,xrange,yrange,zrange,**repulse_kwargs),\
            self._generate_uniform_separated_poses(mdist,N,xrange,yrange,zrange,**repulse_kwargs)
        
        
    def generate_disk_case_with_repulsion(
        self, mdist:float, N:int,
        dist_range:tuple[float,float],
        xrange:tuple[float,float], yrange:tuple[float,float],
        zrange:typing.Optional[tuple[float,float]]=None,
        lat_range:typing.Optional[tuple[float,float]]=None,
        **repulse_kwargs) -> tuple[list[Pose3D],list[Pose3D]]:
        
        og_starts = self._generate_uniform_pts(N,xrange,yrange,zrange)

        if self.verbose > 0:
            print(f"Separating {N} starting points: ",end='')
            
        sep_starts = self._repulse_separate(og_starts,mdist,**repulse_kwargs)
        start_angles = self.rng.uniform(0,2*np.pi,N)
        
        og_ends = self._generate_endpoints_in_disk(sep_starts,dist_range,lat_range)

        if self.verbose > 0:
            print(f"Separating {N} ending points: ",end='')
            
        sep_ends = self._repulse_separate(og_ends,mdist,**repulse_kwargs)
        end_angles = self.rng.uniform(0,2*np.pi,N)
        
        starts  = [Pose3D(p[0],p[1],p[2],a) for p,a in zip(sep_starts,start_angles)]
        ends    = [Pose3D(p[0],p[1],p[2],a) for p,a in zip(sep_ends,end_angles)]
        
        return starts,ends
    
    def generate_random_to_formation(self, mdist:float, formation:Formation,
                                     xrange:tuple[float,float], yrange:tuple[float,float],
                                    zrange:typing.Optional[tuple[float,float]]=None,
                                    **repulse_kwargs) -> tuple[list[Pose3D],list[Pose3D]]:
        
        N = formation.agent_num
        
        og_starts = self._generate_uniform_pts(N,xrange,yrange,zrange)

        if self.verbose > 0:
            print(f"Separating {N} starting points: ",end='')
            
        sep_starts = self._repulse_separate(og_starts,mdist,**repulse_kwargs)
        start_angles = self.rng.uniform(0,2*np.pi,N)
        
        ends_array = formation.get_abs_positions()
        
        starts  = [Pose3D(p[0],p[1],p[2],a) for p,a in zip(sep_starts,start_angles)]
        ends    = [Pose3D(p[0],p[1],p[2],p[3]) for p in ends_array]
        
        return starts,ends
        
    

DEBUG = False

if __name__ == '__main__' and not DEBUG:
    import argparse,time,pathlib,os
    import multiprocessing
    
    parser = argparse.ArgumentParser('Random Path Planning Problem Generator',
                                     description="Generate random path planning problems by sampling points uniformly "
                                     "in a 3D interval, then uses repulsion forces to ensure minimal separation. "
                                     "The end points may be generated like the start points, or based on a distance "
                                     "to their associated starts. Orientations are fully randomized and planar.",
                                     epilog="Usual statistics: Airliner:\n"
                                     " - Airspeed       : 250 knots (4.166 NM/min)\n"
                                     " - Turn radius    : 1.33 NM\n"
                                     " - Min separation : 5 NM\n\n"
                                     "Lab fixed wing:\n"
                                     " - Airspeed       : 15 m/s\n"
                                     " - Turn radius    : 40 m\n"
                                     " - Min separation : 80 m\n\n"
                                     "When generating fully random cases, the minimal test case separation should be "
                                     "the requested minimal separation plus twice the minimal turn radius of each aircraft, "
                                     "as it allows waiting in circle without generating conflicts.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('N',type=int,help='Number of Poses/Aircraft in the case')
    parser.add_argument('mdist',type=float,help='Minimal distance separating aircraft')
    parser.add_argument('output',type=str,help='File prefix name for writing the test case. Will have \'_{test_case_num}.csv\' appended.')
    
    
    parser.add_argument('-s','--seed',dest='seed',
                        type=int, help='Seed for the random number generator. If none, use current time.',default=None)
    
    parser.add_argument('-v','--airspeed',dest='airspeed',
        type=float, help='Airspeed for all aircraft. Default to 1.',default=1.)

    parser.add_argument('-c','--climb',dest='climb',
        type=float, help='Climb speed for all aircraft. Default to 1.',default=1.)

    parser.add_argument('-r','--turn-radius',dest='turn_radius',
        type=float, help='Minimal turn radius for all aircraft. Default to 1.',default=1.)

    
    parser.add_argument('-x','--x-range',dest='x_range',
        nargs=2,help='Low and high values for initial random point generation, X axis. Default to (0,1)',
        default=(0.,1.))

    parser.add_argument('-y','--y-range',dest='y_range',
        nargs=2,help='Low and high values for initial random point generation, Y axis. Default to (0,1)',
        default=(0.,1.))

    parser.add_argument('-z','--z-range',dest='z_range',
        nargs=2,help='Low and high values for initial random point generation, Z axis. Default to (0,0)',
        default=(0.,0.))

    
    parser.add_argument('-d','--d-range',dest='d_range',
                        nargs=2,help='If set, uses disk based generation of endpoints. Specifies low and high values for disk radius.',
                        default=(None,None))
    
    parser.add_argument('-l','--lat-range',dest='lat_range',
                        nargs=2,help='Only relevant if \'--d-range\' is set. Specifies low and high latitudes for disk generation. Default to (0,0)',
                        default=(0,0))
    
    parser.add_argument('-M',type=int,
                        help="Number of test cases to generate (repeat the arguments, increment the seed). Default to 1.",
                        default=1)
    
    parser.add_argument('-p','--processes',dest='processes',
                        type=int,help="Number of processes used to generate cases. Default to 0 (i.e. use as many as possible).",
                        default=0)
    
    
    args = parser.parse_args()
    
    N = abs(args.N)
    mdist = abs(args.mdist)
    processes = abs(args.processes)
    
    xrange = (float(args.x_range[0]),float(args.x_range[1]))
    yrange = (float(args.y_range[0]),float(args.y_range[1]))
    zrange = (float(args.z_range[0]),float(args.z_range[1]))
    
    stats = [ACStats(i,args.airspeed,args.climb,args.turn_radius) for i in range(1,N+1)]
    dts = np.zeros(N)
    seed = int(args.seed) if args.seed is not None else time.time_ns()
    
    path = pathlib.Path(args.output).parent
    if not path.exists():
        os.mkdir(path)
    
    def problem_gen(m:int):
        print(f"Generating case number {m}... ")
        
        gen = RandomPathPlanningGenerator(seed+m,verbose=1)
        
        # Test for use of disk
        if args.d_range[0] is None or args.d_range[1] is None:
            # No disk
            starts,ends = gen.generate_uniform_case_with_repulsion(mdist,N,xrange,yrange,zrange)
            
        else:
            # Disk based generation
            
            drange = (float(args.d_range[0]),float(args.d_range[1]))
            latrange = (float(args.lat_range[0]),float(args.lat_range[1]))
            
            starts,ends = gen.generate_disk_case_with_repulsion(mdist,N,drange,xrange,yrange,zrange,latrange)
        
        filepath = path.joinpath(path.name+f"_{m}.csv")
        print(f"Writing at {filepath}  ...  ",end=' ')
        
        tranposed_problem = [AC_PP_Problem(stat,start,end,dt) for stat,start,end,dt in zip(stats,starts,ends,dts) ]
        
        write_pathplanning_problem_to_CSV(filepath,tranposed_problem)
        
        print("Done!\n")
        
    with multiprocessing.Pool(processes if processes > 0 else None) as p:
        p.map(problem_gen,range(args.M))

    
if __name__ == '__main__' and DEBUG:
    import matplotlib.pyplot as plt
    
    seed = 6
    N = 10
    mdist = 80
    xrange = (0,1000)
    yrange = (0,1000)
    
    gegen = RandomPathPlanningGenerator(seed,verbose=1)
    og_pts = gegen._generate_uniform_pts(N,xrange,yrange)
    final_pts = gegen._repulse_separate(og_pts.copy(),mdist,friction=1.5)
    
    og_mdist = np.sqrt(np.min(pdist(og_pts,'sqeuclid')))
    final_mdist = np.sqrt(np.min(pdist(final_pts,'sqeuclid')))
    
    print(f"Went from\n - mdist: {og_mdist:.3f}\n To\n - mdist: {final_mdist:.3f}")
    
    for i in range(N):
        start = og_pts[i]
        end   = final_pts[i]
        plt.plot([start[0],end[0]],[start[1],end[1]],'k--',)
    
    plt.scatter(og_pts[:,0],og_pts[:,1],c='b',label="Before repulsion")
    plt.scatter(final_pts[:,0],final_pts[:,1],c='r',label="After repulsion")
    
    # og_ends = gegen._generate_endpoints_in_disk(final_pts,(12*mdist,13*mdist))
    # final_ends = gegen._repulse_separate(og_ends.copy(),mdist)
    
    # difficult_ends = gegen._repulse_separate(gegen._generate_uniform_pts(N,xrange,yrange),mdist)
    
    # for i in range(N):
    #     start = final_pts[i]
    #     # tmp_end = og_ends[i]
    #     end   = difficult_ends[i]
    #     # plt.plot([start[0],tmp_end[0],end[0]],[start[1],tmp_end[1],end[1]],'k:',alpha=0.5)
    #     plt.plot([start[0],end[0]],[start[1],end[1]],'k--',)
    
    # plt.scatter(final_pts[:,0],final_pts[:,1],c='b',label="Starting points")
    # # plt.scatter(og_ends[:,0],og_ends[:,1],c='y',label="Ending points (before repulsion)")
    # plt.scatter(difficult_ends[:,0],difficult_ends[:,1],c='r',label="Ending points (after repulsion)")
    
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.show()
    
    