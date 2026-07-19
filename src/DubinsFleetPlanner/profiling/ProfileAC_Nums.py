# Copyright (C) 2026 Mael FEURGARD <mael.feurgard@enac.fr>
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

import typing
from dataclasses import dataclass
import argparse,csv,pathlib,os,copy

from .AllProblemsGenNTest import main as solve_one
from .Formation import quasi_square_upper

@dataclass
class SolveOneArgs:
    """
    Mimic the arguments given to the main in `AllProblemsGenNTest`
    """
    N:int
    mdist:float
    output:str
    basename:str
    solver:str
    
    dist_mult:float     = 1.5
    rng_dist_mult:float = 1.5
    airspeed:float      = 1.
    climb:float         = 1.
    turn_radius:float   = 1.
    
    x_range:tuple[float,float] = (0,1)
    y_range:tuple[float,float] = (0,1)
    xshift:float               = 1.
    
    d_range:tuple[float,float] = (9,11)
    
    wind:tuple[float,float]    = (0,0)
    
    M:int = 100
    
    processes:int = 0
    
    regen:bool  = False
    redo:bool   = False
    skip_sampling:bool = False
    formation_only:bool = False
    rng_only:bool = False
    
# Default values for a case based on 1m-wider fixed-wing UAVs
# Distances are in meters and durations in seconds
# We assume well-separated UAVs
StandardUAVs = SolveOneArgs(
    2,
    80,
    'uavs_2x1',
    'uavs_',
    'DubinsFleetPlanner',
    airspeed=15,
    turn_radius=40,
    dist_mult=1.5,
    rng_dist_mult=3.,
    x_range=(0,1000),
    y_range=(0,1000),
    xshift = 1000,
    M=100
)

# Variation of the previous one, but assume tight formations,
# i.e. the aircraft separation is small compared to their turn radius
TightUAVs = SolveOneArgs(
    2,
    10,
    'uavs_tight_2x1',
    'uavs_tight_',
    'DubinsFleetPlanner',
    airspeed=15,
    turn_radius=40,
    dist_mult=1.5,
    rng_dist_mult=3.,
    x_range=(0,1000),
    y_range=(0,1000),
    xshift = 1000,
    M=100
)

# Default values for a case based on commercial airliners
# Distances are in Nautical Miles and durations in minutes
# (Speed is 250 knots, converted in NM/min; 
# Turn radius is equivalent to a full circle in 2 minutes)
StandardAirplanes = SolveOneArgs(
    5,
    5,
    'airplanes_5x1',
    'airplanes_',
    'DubinsFleetPlanner',
    airspeed=4.166,
    turn_radius=1.33,
    dist_mult=1.5,
    rng_dist_mult=3.,
    x_range=(0,100),
    y_range=(0,100),
    xshift = 100,
    M=100
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        'Meta Benchmark launcher'
    )
    
    parser.add_argument('ac',choices=['UAV_std','UAV_tight','AC_std'],
                        help="Type of aircraft and formation")
    parser.add_argument('Ns',nargs=2,type=int,
                        help="Min and max number of aircraft for testing formations")
    parser.add_argument('output',
                        help='Folder where cases and solutions should be put')
    parser.add_argument('solver',
                        help='Location of the DubinsFleetPlanner solver')
    
    parser.add_argument('-M',type=int, default=100,
                        help='Number of random cases to do per formation. Default to 100.')
    
    parser.add_argument('-p','--processes',dest='processes',
                        type=int,help="Number of processes used to generate cases and solve problems. Default to 0 (i.e. use as many as possible).",
                        default=0)
    
    parser.add_argument('-w','--wind',dest='wind', nargs=2,
                        help='XY Wind, as a pair of values (X,Y). Default to (0,0), i.e. no wind.',
                        default=(0,0))
    
    parser.add_argument('--regen',dest='regen',action='store_true',
                        help="Flag. If set, force regenerating the test cases (but may still skip identical cases).\
                        Default to False.",
                        default=False)
    
    parser.add_argument('--redo',dest='redo',action='store_true',
                        help="Flag. If set, force regenerating and restarting the solver on every directory.\
                        Otherwise, skip the directories where a complete 'summary.csv' file is found. Default to False.",
                        default=False)
    
    parser.add_argument('--skip-sampling',dest='skip_sampling',action='store_true',
                        help="Flag. If set, disable checking separation in result files. Default to False",
                        default=False)
    
    args = parser.parse_args()
    
    output_dir = pathlib.Path(args.output)
    os.makedirs(output_dir,exist_ok=True)
    
    if args.ac == 'UAV_std':
        basecase = StandardUAVs
    elif args.ac == 'UAV_tight':
        basecase = TightUAVs
    elif args.ac == 'AC_std':
        basecase = StandardAirplanes
    else:
        raise ValueError(f"Unknown formation type: {args.ac}")
    

    for n in range(int(args.Ns[0]),int(args.Ns[1])+1):
        
        this_case = copy.deepcopy(basecase)
        this_case.N      = n
        ncols,nlines     = quasi_square_upper(n)
        print(n,ncols,nlines)
        this_case.output = str((output_dir / (this_case.basename+str(n)+"-"+str(ncols)+"x"+str(nlines))).resolve())
        this_case.solver = args.solver
        
        this_case.processes = args.processes
        this_case.wind      = (float(args.wind[0]),float(args.wind[1]))
        this_case.regen     = args.regen
        this_case.redo      = args.redo
        this_case.skip_sampling = args.skip_sampling
        
        print("========== ========== ========== ========== ========== ==========")
        solve_one(this_case)
            
            