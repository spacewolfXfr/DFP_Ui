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

from .Poses import Pose3D,poses_dist,min_XY_dist

from .profiling.ProblemGenerator import RandomPathPlanningGenerator,ACStats,AC_PP_Problem

from .profiling.FormationProblemGenerator import list_all_formations,generate_basic_formation_moves,generate_formation_transitions,quasi_square_upper
    
from .ioUtils import CaseSummary,parse_result_summary,write_summary,parse_trajectories_from_JSON,\
    parse_pathplanning_problem_from_CSV,write_pathplanning_problem_to_CSV
    
import numpy as np
import typing,os,subprocess,pathlib,multiprocessing,time,hashlib

def count_testcases_in_dir(case_dir:pathlib.Path) -> int:
    return sum(1 for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() == '.csv')
     

def write_or_skip_CSV_case(file,data:typing.Sequence[AC_PP_Problem],overwrite:bool=False):
    try:
        write_pathplanning_problem_to_CSV(file,data,overwrite)
    except FileExistsError:
        pass

def check_result(test_src:pathlib.Path,result_file:pathlib.Path,summary:typing.Optional[CaseSummary],tol:float=1e-2) -> bool:
    """ Check that the given files match

    Args:
        test_src (pathlib.Path): Path for the test case source
        result_file (pathlib.Path): Path for the test case result
        summary (typing.Optional[CaseSummary]): Summary of results
        tol (float, optional): Tolerance for checking separation; disabled if negative. Defaults to 1e-2.

    Returns:
        bool: _description_
    """
    print(f"Checking {result_file.name}")
    
    ## Parsing from files
    pb = parse_pathplanning_problem_from_CSV(test_src)
    try:
        sol = parse_trajectories_from_JSON(result_file)
    except FileNotFoundError:
        print(f"{result_file} was not found")
        return False
    
    ## Check that AC ids match
    keys = set(sol.list_ids())
    
    try:
        sol_starts  = sol.starts
    except Exception as e:
        print(f"Error when computing solution starts of {result_file}:\n{e}")
        return False
    
    if not(sol_starts.keys() <= keys):
        print(f"{result_file} has an inconsistent number of aircraft!")
        return False
    
    try:
        sol_ends    = sol.ends
    except Exception as e:
        print(f"Error when computing solution ends:\n{e}")
        return False
    
    if not(sol_ends.keys() <= keys):
        print(f"{result_file} has an inconsistent number of aircraft!")
        return False
    
    pb_starts   = dict()
    pb_ends     = dict()
    
    for l in pb:
        pb_starts[l.stats.id] = l.start
        pb_ends[l.stats.id] = l.end
        
    if not(pb_starts.keys() <= keys):
        print(f"{test_src} has an inconsistent number of aircraft!")
        return False
    
    if not(pb_ends.keys() <= keys):
        print(f"{test_src} has an inconsistent number of aircraft!")
        return False
    
    ## Check that endpoint matches
    for id in keys:
        sol_s   = sol_starts[id]
        sol_e   = sol_ends[id]
        pb_s    = pb_starts[id]
        pb_e    = pb_ends[id]
        
        if poses_dist(sol_s,pb_s) > tol:
            print(f"{test_src} starts do not match the associated solution!\n\t{sol_s.to_numpy()} VS {sol_e.to_numpy()}")
            return False
        if poses_dist(sol_e,pb_e) > tol:
            print(f"{test_src} ends do not match the associated solution!\n\t{sol_e.to_numpy()} VS {pb_e.to_numpy()}")
            return False
    
    
    ## Check separation
    if summary is None:
        print(f"No summary entry found for {test_src}")
        return False
    
    if summary.success and not(summary.false_positive):
        sep = sol.separation
        
        # Skip validation if tol is negative
        if tol < 0:
            return True
        
        samples = sol.sample_poses(int(sol.duration/tol))
        
        for _,id_poses in samples:
            d,_,_ = min_XY_dist(list(id_poses.values()))
            if d < sep:
                print(f"False positive detected in {result_file}")
                return False
        
    return True


def check_results(case_dir:pathlib.Path,sol_dir:pathlib.Path,skip_sampling:bool=False) -> set[pathlib.Path]:
    todos = set(p.resolve() for p in case_dir.iterdir() if p.is_file() and p.suffix.lower() == '.csv')

    summary_file = sol_dir / "summary.csv"
    if not(summary_file.exists()):
        for test_file in todos:
            filestem = test_file.stem
            sol_file = sol_dir / (filestem + ".sol.json")
            try:
                os.remove(sol_file)
            except FileNotFoundError:
                pass
        return todos
    
    summary_db = parse_result_summary(summary_file)
    
    
    filechecking_args = []
    for file in todos:
        filestem = file.stem
        sol_file = sol_dir / (filestem + ".sol.json")
        
        filechecking_args.append(
            (file,sol_file,summary_db.get(str(file.resolve())), -1 if skip_sampling else 1e-2)
        )
    
    redo = set()
    ret = [check_result(*t) for t in filechecking_args]
        
    for test_file,valid in zip(todos,ret):
        
        if valid:
            continue
        
        filestem = test_file.stem
        sol_file = sol_dir / (filestem + ".sol.json")
        
        try:
            del summary_db[str(test_file)]
        except KeyError:
            pass
        
        try:
            os.remove(sol_file)
        except FileNotFoundError:
            pass
        redo.add(test_file)
        
    
    write_summary(summary_file,summary_db)
    
    if None in redo:
        redo.remove(None)
    
    return redo
    
    
        

def solve_problems(solver:pathlib.Path,src_dir:pathlib.Path,dest_dir:pathlib.Path,
                   separation:float, wind:tuple[float,float],threads:int,
                   **kwargs) -> subprocess.CompletedProcess[bytes]:
    cmd = []
    cmd.append(str(solver.resolve()))
    cmd.append(str(src_dir.resolve()))
    cmd.append(str(dest_dir.resolve()))
    cmd.append(str(separation))
    
    cmd.append(str(wind[0]))
    cmd.append(str(wind[1]))
    
    cmd.append('-t')
    cmd.append(str(threads))
    
    for k,v in kwargs.items():
        cmd.append(k)
        
        if v is None:
            continue
        
        cmd.append(str(v))
    
    return subprocess.run(cmd)

def main(args):
    ## Gathering parsed values and closely related
    
    # Number of aircraft and formation config
    N = args.N
    ncols,nlines = quasi_square_upper(N)
    
    # Minimal separation
    mdist = args.mdist
    assert args.dist_mult >= 1.
    endpoint_dist = mdist * args.dist_mult
    if args.rng_dist_mult is None:
        rng_endpoint_dist = endpoint_dist
    else:
        assert args.rng_dist_mult >= 1.
        rng_endpoint_dist = mdist * args.rng_dist_mult
    
    # test cases and solutions locations
    dir = pathlib.Path(args.output)
    
    rng_pb_dir                 = dir / "rng" / "cases"
    os.makedirs(rng_pb_dir,exist_ok=True)
    formation_pb_dir           = dir / "formation" / "cases"
    os.makedirs(formation_pb_dir,exist_ok=True)
    rng_to_formation_pb_dir    = dir / "rng_to_formation" / "cases"
    os.makedirs(rng_to_formation_pb_dir,exist_ok=True)
    
    rng_sol_dir                 = dir / "rng" / "solutions"
    os.makedirs(rng_sol_dir,exist_ok=True)
    formation_sol_dir           = dir / "formation" / "solutions"
    os.makedirs(formation_sol_dir,exist_ok=True)
    rng_to_formation_sol_dir    = dir / "rng_to_formation" / "solutions"
    os.makedirs(rng_to_formation_sol_dir,exist_ok=True)
    
    # Aircraft stats
    airspeed    = args.airspeed
    climb       = args.climb
    turn_radius = args.turn_radius
    
    stats = [ACStats(i,airspeed,climb,turn_radius) for i in range(N)]
    
    # Random case generation parameters
    x_range = (float(args.x_range[0]),float(args.x_range[1]))
    y_range = (float(args.y_range[0]),float(args.y_range[1]))
    z_range = (0,0)
    xshift = args.xshift
    
    midpoint = Pose3D(
        (x_range[0]+x_range[1])/2,
        (y_range[0]+y_range[1])/2,
        (z_range[0]+z_range[1])/2,
        0.)
    
    d_range = (float(args.d_range[0]),float(args.d_range[1]))
    M = args.M
    
    # Solver parameters
    processes   = args.processes
    solver      = args.solver
    wind        = args.wind
    
    if args.redo:
        args.regen = True
    
    seed = abs(int(hashlib.sha1(str(args.output).encode()).hexdigest(),base=16))
    
    print(f"The seed is: {seed}\n\n")
    print(f"Formation: {ncols} x {nlines}")
        
        
    # Test to performs
    
    test_formations = True
    test_rng        = True
    test_rng_to_formation = True

    if args.formation_only and args.rng_only:
        raise ValueError("Cannot set both 'formation-only' and 'rng-only' flags simultaneously")

    if args.rng_only:
        test_formations = False
    if args.formation_only:
        test_rng = False
        test_rng_to_formation = False


    ## Generating formation poses
    dts = np.zeros(N)
    
    all_formations = list_all_formations(ncols,nlines,N,endpoint_dist)
    formations_std_poses:list[list[Pose3D]] = []
    formations_shifted_poses:list[list[Pose3D]] = []
    for f in all_formations:
        f = f.apply_rotation().to_barycentric_coords()
        
        fname = f.name.lower()
        if 'circle' in fname:
            pass
        else:
            if 'rectangle' in fname or 'diamond' in fname or 'stacked' in fname:
                pass
            else:
                f = f.sort_y()
        
        f.center[0] = midpoint.x
        f.center[1] = midpoint.y
        f.center[2] = midpoint.z
        endposes = f.get_abs_positions()
        
        shifted_endposes = endposes.copy()
        shifted_endposes[:,0] += xshift
        
        formations_std_poses.append(
            [Pose3D.from_array(a) for _,a in zip(range(N),endposes)]
        )
        formations_shifted_poses.append(
            [Pose3D.from_array(a) for _,a in zip(range(N),shifted_endposes)]
        )
        
        f.center = np.zeros(3)
    
    ## Formation problems
    
    # formation_depl = endpoint_dist * N
    
    if test_formations:
        for f in all_formations:
            pb_list = generate_basic_formation_moves(f,stats,xshift)
            
            for pb,turn_name in pb_list:
                write_or_skip_CSV_case(formation_pb_dir / (f.name+"_"+turn_name+".csv"),pb,args.redo)
                
        for pb,name in generate_formation_transitions(all_formations,stats,xshift):
            write_or_skip_CSV_case(formation_pb_dir / (name+".csv"),pb,args.redo)
        
    ## Random problems
    
    def problem_gen(m:int):
        print(f"Generating case number {m}... ")
        
        gen = RandomPathPlanningGenerator(seed+m,verbose=0)
        
        og_pts = gen._generate_uniform_pts(N,x_range,y_range,z_range)
        
        sep_pts = gen._repulse_separate(og_pts,rng_endpoint_dist)
        angles = gen.rng.uniform(0,2*np.pi,N)
        
        rng_starts = [Pose3D(p[0],p[1],p[2],a) for p,a in zip(sep_pts,angles)]
        
        # Fully random to fully random poses (2D) and with away variant
        
        rng_file = rng_pb_dir / f"rng_rng_{m}.csv"
        rng_away_file = rng_pb_dir / f"rng_rng_away_{m}.csv"
        
        
        if test_rng and args.regen or not(rng_file.exists()) or not(rng_away_file.exists()):
            end_pts = gen._generate_uniform_pts(N,x_range,y_range,z_range)
            
            sep_end_pts = gen._repulse_separate(end_pts,rng_endpoint_dist)
            end_angles = gen.rng.uniform(0,2*np.pi,N)
            
            rng_ends = [Pose3D(p[0],p[1],p[2],a) for p,a in zip(sep_end_pts,end_angles)]
            rng_problem = [AC_PP_Problem(stat,start,end,dt) for stat,start,end,dt in zip(stats,rng_starts,rng_ends,dts) ]
            
            write_pathplanning_problem_to_CSV(rng_file,rng_problem,args.regen)
            
            
            rng_shifted_ends = [Pose3D(p[0]+xshift,p[1],p[2],a) for p,a in zip(sep_end_pts,end_angles)]
            rng_shifted_problem = [AC_PP_Problem(stat,start,end,dt) for stat,start,end,dt in zip(stats,rng_starts,rng_shifted_ends,dts) ]

            write_pathplanning_problem_to_CSV(rng_away_file,rng_shifted_problem,args.regen)
        
        # Fully random to reasonnably close random poses (2D)
        
        drange = (float(d_range[0]),float(d_range[1]))
        latrange = (0.,0.)
        
        disk_file = rng_pb_dir / f"rng_disk_{m}.csv"
        if test_rng and args.regen or not(disk_file.exists()):
            dlow = endpoint_dist * drange[0]
            dhigh = endpoint_dist * drange[1]
            disk_og_ends = gen._generate_endpoints_in_disk(sep_pts,(dlow,dhigh),latrange)

            disk_sep_ends = gen._repulse_separate(disk_og_ends,rng_endpoint_dist)
            disk_end_angles = gen.rng.uniform(0,2*np.pi,N)
            
            disk_ends = [Pose3D(p[0],p[1],p[2],a) for p,a in zip(disk_sep_ends,disk_end_angles)]
            
            disk_problem = [AC_PP_Problem(stat,start,end,dt) for stat,start,end,dt in zip(stats,rng_starts,disk_ends,dts) ]

                
            write_pathplanning_problem_to_CSV(disk_file,disk_problem,args.regen)
        
        # Fully random to formation (2D)
        if test_rng_to_formation:
            for f,f_ends in zip(all_formations,formations_std_poses):
                f_problem = [AC_PP_Problem(stat,start,end,dt) for stat,start,end,dt in zip(stats,rng_starts,f_ends,dts) ]

                f_file = rng_to_formation_pb_dir / f"rng_to_{f.name}_{m}.csv"
                
                write_or_skip_CSV_case(f_file,f_problem,args.regen)
            
            for f,f_shiftend_ends in zip(all_formations,formations_shifted_poses):
                f_problem = [AC_PP_Problem(stat,start,end,dt) for stat,start,end,dt in zip(stats,rng_starts,f_shiftend_ends,dts) ]
                    
                f_file = rng_to_formation_pb_dir / f"rng_to_{f.name}_away_{m}.csv"
                
                write_or_skip_CSV_case(f_file,f_problem,args.regen)
            
        print(f"Done with {m}")
    
    if test_rng or test_rng_to_formation:
        for i in range(M):
            problem_gen(i)
        # with multiprocessing.Pool(processes if processes > 0 else None) as p:
            # p.map(problem_gen,range(args.M))
        
    ## Solving
    
    extra_args = {'-l':None,'-v':'1'}
    if args.redo:
        extra_args['-R'] = None
    
    if solver is not None:
        solver = pathlib.Path(solver)
        
        if test_formations:
            if not args.redo:
                failed_cases = check_results(formation_pb_dir,formation_sol_dir,args.skip_sampling)
                formation_done = len(failed_cases) == 0
            else:
                formation_done = False
            if args.redo or not(formation_done):
                now = time.time()
                solve_problems(solver,formation_pb_dir,formation_sol_dir,mdist,wind,processes,**extra_args)
                dt = time.time() - now
                
                print("\n\n")
                print("===============================================")
                print("         Done with formations problems       ")
                print(f"             Time spent: {dt:.3f}s           ")        
                print("===============================================\n\n")
                
            else:
                print(f" === Formations problems checked and done ===")
        else:
            print(f" ~~~ Formations problems disabled ~~~")
            
        
        
        
        if test_rng:
            if not args.redo:
                failed_cases = check_results(rng_pb_dir,rng_sol_dir,args.skip_sampling)
                rng_done = len(failed_cases) == 0
            else:
                rng_done = False
            if args.redo or not(rng_done):
                now = time.time()
                solve_problems(solver,rng_pb_dir,rng_sol_dir,mdist,wind,processes,**extra_args)
                dt = time.time() - now
                
                print("\n\n")
                print("===============================================")
                print("             Done with RNG problems          ")
                print(f"             Time spent: {dt:.3f}s           ")
                print("===============================================\n\n")
            else:
                print(f" === RNG problems checked and done ===")
        else:
            print(f" ~~~ RNG problems disabled ~~~")
        
        
        
        if test_rng_to_formation:
            if not args.redo:
                failed_cases = check_results(rng_to_formation_pb_dir,rng_to_formation_sol_dir,args.skip_sampling)
                rng_to_formation_done = len(failed_cases) == 0
            else:
                rng_to_formation_done = False
            if args.redo or not(rng_to_formation_done):
                now = time.time()
                solve_problems(solver,rng_to_formation_pb_dir,rng_to_formation_sol_dir,mdist,wind,processes,**extra_args)
                dt = time.time() - now
                
                print("\n\n")
                print("===============================================")
                print("      Done with RNG to formation problems    ")
                print(f"             Time spent: {dt:.3f}s           ")        
                print("===============================================\n\n")
            else:
                print(f" === RNG to Formation problems checked and done ===")
        else:
            print(f" ~~~ RNG to Formation problems disabled ~~~")

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser('Formation Path Planning Problem Generator',
                                     description="Generate formation-based path planning problems from a list of known formations."
                                     "The end points may be generated by shifting and rotating the initila formation, "
                                     "or transitioning from a formation to another",
                                     epilog="Usual statistics: Airliner:\n"
                                     " - Airspeed       : 250 knots (4.166 NM/min)\n"
                                     " - Turn radius    : 1.33 NM (full circle in 2 minutes)\n"
                                     " - Min separation : 5 NM\n\n"
                                     "Lab fixed wing:\n"
                                     " - Airspeed       : 15 m/s\n"
                                     " - Turn radius    : 40 m\n"
                                     " - Min separation : 80 m\n\n",
                                     formatter_class=argparse.RawTextHelpFormatter)
    
    parser.add_argument('N',type=int,help='Number of aicraft')
    parser.add_argument('mdist',type=float,help='Minimal distance separating aircraft')
    parser.add_argument('output',type=str,help='Destination folder for test cases (and the eventual solutions).')
    
    parser.add_argument('-m','--dist-mult',dest='dist_mult',
                        type=float,help='Multiplier for mdist for setting the formation separation. Should be larger than 1. Default to 1.5',
                        default=1.5)
    
    parser.add_argument('--rng-dist-mult',dest='rng_dist_mult',
                        type=float,help='Multiplier for mdist for setting the RNG separation. Should be larger than 1. Default to being equal to dist-mult',
                        default=None)
    
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
    
    parser.add_argument('--xshift',dest='xshift',
        type=float,help='X shift value for AWAY cases. Default to 1.',
        default=1.)

    
    parser.add_argument('-d','--d-range',dest='d_range',
                        nargs=2,help='Disk based generation of endpoints. Specifies low and high values for disk radius as a multiple of turn radius.'
                        'Default to (9,11)',
                        default=(9,11))

    
    parser.add_argument('-M',type=int,
                        help="Number of test cases to generate (repeat the arguments, increment the seed). Default to 100.",
                        default=100)
    
    parser.add_argument('-p','--processes',dest='processes',
                        type=int,help="Number of processes used to generate cases and solve problems. Default to 0 (i.e. use as many as possible).",
                        default=0)
    
    parser.add_argument('-s','--solver',dest='solver',
                        help='If defined, path to the solver. In that case, it will be automatically called to solve the generated cases',
                        default=None)
    
    parser.add_argument('-w','--wind',dest='wind', nargs=2,
                        help='XY Wind, as a pair of values (X,Y). Default to (0,0), i.e. no wind.',
                        default=(0,0))
    
    parser.add_argument('--regen',dest='regen',action='store_true',
                        help="Flag. If set, force regenerating the test cases (but may still skip identical cases). Default to False.",
                        default=False)
    
    parser.add_argument('--redo',dest='redo',action='store_true',
                        help="Flag. If set, force regenerating and restarting the solver on every directory. Otherwise, skip the directories where a complete 'summary.csv' file is found. Default to False.",
                        default=False)
    
    parser.add_argument('--skip-sampling',dest='skip_sampling',action='store_true',
                        help="Flag. If set, disable checking separation in result files. Default to False",
                        default=False)
    
    parser.add_argument('--formation-only',dest='formation_only',action='store_true',
                        help="Flag. If set, generate and test only deterministic formation cases",
                        default=False)
    
    parser.add_argument('--rng-only',dest='rng_only',action='store_true',
                        help="Flag. If set, generate and test only randomized cases",
                        default=False)
    
    main(parser.parse_args())
            
        