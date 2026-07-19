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

import typing
import json
from dataclasses import dataclass
from operator import itemgetter

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import numpy as np

from .ioUtils import parse_trajectories_from_JSON

@dataclass
class RichConflict:
    AC_id1:int
    AC_id2:int
    Path_type1:str
    Path_type2:str
    min_loc:float
    min_val:float
    
def parse_from_json(file:str) -> list[tuple[float,list[RichConflict]]]:
    output = []
    with open(file) as fp:
        d = json.load(fp)
        for e in d:
            conflicts = [RichConflict(**v) for v in e["conflicts"]]
            output.append(
                (e["time"],conflicts)
            )
    
    return output

# For each situation represented by (ac_id1,path_type1,ac_id2,path_type2),
# give the list of conflicts, each represented by (time,min_dist_loc,min_dist_val)
ConflictDict = dict[tuple[int,str,int,str],list[tuple[float,float,float]]]

def tranpose_conflict_list(list_of_conflicts:list[tuple[float,list[RichConflict]]]) \
    -> ConflictDict:
    
    output:dict[tuple[int,str,int,str],list[tuple[float,float,float]]] = {}
    for time,conflict_list in list_of_conflicts:
        for r in conflict_list:
            try:
                output[(r.AC_id1,r.Path_type1,r.AC_id2,r.Path_type2)].append((time,r.min_loc,r.min_val))
            except KeyError:
                output[(r.AC_id1,r.Path_type1,r.AC_id2,r.Path_type2)] = [(time,r.min_loc,r.min_val)]
    return output

def filter_conflicts(d:ConflictDict, maximum_value:float) -> ConflictDict:
    for k,v in d.items():
        fresh_v = list(filter(lambda t: t[2] < maximum_value, v))
        d[k] = fresh_v 
    return d

def plot_conflict_lines(ax:Axes,d:ConflictDict) -> tuple[Axes,dict[tuple[int,str,int,str],float]]:
    labels = []
    yi = 0
    
    max_sep = 0.
    for v in d.values():
        if len(v) == 0:
            continue
        else:
            max_sep = max(max_sep,max(v,key=itemgetter(2))[2])
    
    label_dict:dict[tuple[int,str,int,str],float] = {}
    
    for k,v in d.items():
        if len(v) == 0:
            continue
        this_xs = [e[0] for e in v]
        this_ys = [yi*1.2*max_sep + e[2] for e in v]
        
        labels.append(k)
        yi += 1
        
        ax.scatter(this_xs,this_ys)
        label_dict[k] = yi*1.2*max_sep
    
    
    ax.set_yticks(list(v for v in label_dict.values()),list(str(k) for k in label_dict.keys()))
    
    return ax,label_dict
    

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser("Conflicts parser and plotter")
    
    parser.add_argument('file',type=str,help="JSON file containing the conflicts")
    parser.add_argument('-s','--separation', type=float,dest="sep",
                        help="Minimal separation value for declaring a conflict. If left undefined, don't filter",
                        default=None)
    parser.add_argument('-r','--result',dest='result',
                        help="Optional, file containing the path planning solution (in JSON format); enrich display",
                        default=None)
    
    args = parser.parse_args()
    
    conflicts = tranpose_conflict_list(parse_from_json(args.file))
    
    fig,ax = plt.subplots()
    
    if args.sep is not None:
        ax,label_dict = plot_conflict_lines(ax,filter_conflicts(conflicts,args.sep))
    else:
        ax,label_dict = plot_conflict_lines(ax,conflicts)
        
    if args.result is not None:
        fleet_plan = parse_trajectories_from_JSON(args.result)
        
        ax.vlines(fleet_plan.duration,ax.get_ylim()[0],ax.get_ylim()[1],label=f'Solution duration: {fleet_plan.duration:.3f}',colors='red')
        
        xmin,xmax = ax.get_xlim()
        
        first_label = True
        
        for s1,p1 in fleet_plan.trajectories:
            for s2,p2 in fleet_plan.trajectories:
                t_conflict = (s1.id,p1.abbr(),s2.id,p2.abbr())
                if t_conflict in label_dict.keys():
                    if first_label:
                        ax.hlines(label_dict[t_conflict]+fleet_plan.separation,xmin,xmax,color='black',label="Present in solution")
                        first_label = False
                    else:
                        ax.hlines(label_dict[t_conflict]+fleet_plan.separation,xmin,xmax,color='black')
        
        
    
    ax.legend()
    plt.show()
    
    