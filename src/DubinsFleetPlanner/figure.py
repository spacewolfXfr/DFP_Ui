#!/usr/bin/python3

import typing

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple


from .Dubins import LRL,RLR,LSL,LSR,RSL,RSR,SLS,SRS, DubinsMove,Pose2D,Pose3D,ACStats,Path,poses_XY_dist
from .DubinsPathFitting import plan_LRL, plan_RLR, plan_LSL, plan_LSR, plan_RSL, plan_RSR, plan_SLS, plan_SRS,\
        plan_LRL_from_straights, plan_RLR_from_straights, plan_LSL_from_straights, plan_LSR_from_straights,\
        plan_RSL_from_straights, plan_RSR_from_straights, plan_SLS_from_straights, plan_SRS_from_straights,\
        fit_LRL_radius, fit_RLR_radius, fit_LSL_radius, fit_LSR_radius, fit_RSL_radius, fit_RSR_radius, fit_SLS_radius, fit_SRS_radius
from .UI.plotting import plot_pose2d_sequence

from .PathInterpolator import interpolate_path,unit_interpolate_basicpath

from .UI._plotting_extra import ColorType, my_cmap6,my_cmap
 

def LRL_length(start:Pose2D, end:Pose2D) -> float:
    ans = LRL(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def RLR_length(start:Pose2D, end:Pose2D) -> float:
    ans = RLR(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def LSL_length(start:Pose2D, end:Pose2D) -> float:
    ans = LSL(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def LSR_length(start:Pose2D, end:Pose2D) -> float:
    ans = LSR(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)

def RSL_length(start:Pose2D, end:Pose2D) -> float:
    ans = RSL(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def RSR_length(start:Pose2D, end:Pose2D) -> float:
    ans = RSR(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def SLS_length(start:Pose2D, end:Pose2D) -> float:
    ans = SLS(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def SRS_length(start:Pose2D, end:Pose2D) -> float:
    ans = SRS(start,end)
    if ans is None:
        return np.nan
    else:
        return sum(ans)
    
def all_lengths(start:Pose2D, end:Pose2D) -> np.ndarray:
    return np.array([
        LRL_length(start,end),
        RLR_length(start,end),
        LSL_length(start,end),
        LSR_length(start,end),
        RSL_length(start,end),
        RSR_length(start,end),
        SLS_length(start,end),
        SRS_length(start,end),
    ])
    
def compute_for_different_radius(rs:np.ndarray,start:Pose2D,end:Pose2D) -> np.ndarray:
    output = []
    for r in rs:
        sx = start[0]/r
        sy = start[1]/r
        
        ex = end[0]/r
        ey = end[1]/r
        
        output.append(r*all_lengths(Pose2D(sx,sy,start[2]),Pose2D(ex,ey,end[2])))
    
    return np.asarray(output)

def compute_for_different_shifts(radius:float,lengths:np.ndarray,start:Pose2D,end:Pose2D,ratio:float) -> np.ndarray:
    output = []
    
    vsx = np.cos(start.angle)
    vsy = np.sin(start.angle)
    
    vex = np.cos(end.angle)
    vey = np.sin(end.angle)
    
    
    for l in lengths:
        sx = (start[0] + vsx*l*ratio)/radius
        sy = (start[1] + vsy*l*ratio)/radius
        
        ex = (end[0] - vex*l*(1-ratio))/radius
        ey = (end[1] - vey*l*(1-ratio))/radius
    
        output.append(l+radius*all_lengths(Pose2D(sx,sy,start.angle),Pose2D(ex,ey,end.angle)))
        
    return np.asarray(output)


#################### Plotting ####################

def plot_fits(stats:ACStats, start:Pose2D, end:Pose2D):
    fig,axe = plt.subplots()
    axe:Axes
    
    j = 0
    
    colors = [my_cmap6[0],my_cmap6[2],my_cmap6[3],my_cmap6[4]]
    
    for name,planner,marker,markersize in [#("LRL",plan_LRL),
                         #("RLR",plan_RLR), 
                         ("LSL",plan_LSL,"o",5),
                         #("LSR",plan_LSR), 
                         #("RSL",plan_RSL),
                         ("RSR",plan_RSR,"+",11), 
                         ("SLS",plan_SLS,"o",5), ("SRS",plan_SRS,"+","11")
                         ]:
        
        color = colors[ j % len(colors)]
        # color = '0.2'
        j += 1
        
        path = planner(stats,start,end)
        if path is not None:
            l = path.total_length
            samples = np.linspace(0,l,100)
            points = []
            
            for i,s in enumerate(samples):
                points.append(path.pose_at(samples[i]))
            
            
            plot_pose2d_sequence(axe,points,False,True,label=f"S-{name}",
                color=color, 
                marker=marker,markersize=markersize,markevery=0.2,
                linestyle='dashed')
        
    for spine in axe.spines:
        axe.spines[spine].set_visible(False)
        
    axe.set_xticks([])
    axe.set_yticks([])
    axe.set_aspect('equal')
    axe.legend()
    plt.show()


def plot_straights(stats:ACStats, start:Pose2D, end:Pose2D, target_length:float):
    fig,axe = plt.subplots()
    axe:Axes
    
    j = 0
    
    for name,planner,marker,markersize in [#("LRL",plan_LRL_from_straights),
                         #("RLR",plan_RLR_from_straights), 
                         ("LSL",plan_LSL_from_straights,"o",5),
                         #("LSR",plan_LSR_from_straights), 
                         #("RSL",plan_RSL_from_straights),
                         ("RSR",plan_RSR_from_straights,"+",11), 
                         #("SLS",plan_SLS_from_straights), ("SRS",plan_SRS_from_straights)
                         ]:
        color = my_cmap6[j % len(my_cmap6)]
        # color = '0.2'
        j += 2
        
        path = planner(stats,start,end, target_length,1)
        if path is not None:
            l = path.total_length
            samples = np.linspace(0,l,100)
            points = []
            
            for i,s in enumerate(samples):
                points.append(path.pose_at(samples[i]))
            
            
            plot_pose2d_sequence(axe,points,False,True,label=f"S-{name}",
                color=color, 
                marker=marker,markersize=markersize,markevery=0.2,
                linestyle='dashed')
            
        path = planner(stats,start,end, target_length,0)
        if path is not None:
            l = path.total_length
            samples = np.linspace(0,l,100)
            points = []
            
            for i,s in enumerate(samples):
                points.append(path.pose_at(samples[i]))
            
            
            plot_pose2d_sequence(axe,points,False,True,label=f"{name}-S",
                color=color, 
                marker=marker,markersize=markersize,markevery=0.2,
                linestyle='dotted')
            
        path = planner(stats,start,end, target_length,0.5)
        if path is not None:
            l = path.total_length
            samples = np.linspace(0,l,100)
            points = []
            
            for i,s in enumerate(samples):
                points.append(path.pose_at(samples[i]))
            
            
            plot_pose2d_sequence(axe,points,False,True,label=f"S-{name}-S",
                color=color, 
                marker=marker,markersize=markersize,markevery=0.2,
                linestyle='solid')
        
    for spine in axe.spines:
        axe.spines[spine].set_visible(False)
        
    axe.set_xticks([])
    axe.set_yticks([])
    axe.set_aspect('equal')
    axe.legend()
    plt.show()

def plot_length_functions(stats:ACStats, d:float, start:Pose2D, end:Pose2D, step, plot_width, target_lengths):
    bins = np.arange(target_lengths[0],target_lengths[1]+step,step)
    centers = (bins[0:-1] + bins[1:])/2
    
    
    radius_samples  = np.linspace(stats.turn_radius,target_lengths[1],3000,True)
    radius_based    = compute_for_different_radius(radius_samples,
                                                start,end)
    
    length_samples = np.linspace(0,target_lengths[1],3000,True)
    both_shift  = compute_for_different_shifts(stats.turn_radius,length_samples,start,end,0.5)
    end_shift   = compute_for_different_shifts(stats.turn_radius,length_samples,start,end,0)
    first_shift = compute_for_different_shifts(stats.turn_radius,length_samples,start,end,1)
    
    list_samples= [length_samples,length_samples,length_samples,radius_samples]
    list_data   = [first_shift,end_shift,both_shift,radius_based]
    
    titles      = ["Shift start", "Shift end", "Shift both", "Turn radius"]
    markers     = [5,4,7,6]
    colors      = [my_cmap6[0],my_cmap6[2],my_cmap6[4], my_cmap6[5]]
                   
    fig,axes = plt.subplots(2,2,sharey=True)
    
    
    for i,(samples,values,title) in enumerate(zip(list_samples,
                                                  list_data,
                                                   titles)):
        ax:Axes = axes[i % 2][i // 2]
        if i == 0:
            ax.set_xlabel("Radius")
        else:
            ax.set_xlabel("Straigh extension")

        
        ax.plot(samples,values[:,0],label='LRL', color=my_cmap[0])
        ax.plot(samples,values[:,1],label='RLR', color=my_cmap[1])
        ax.plot(samples,values[:,2],label='LSL', color=my_cmap[2])
        ax.plot(samples,values[:,3],label='LSR', color=my_cmap[3])
        ax.plot(samples,values[:,4],label='RSL', color=my_cmap[4])
        ax.plot(samples,values[:,5],label='RSR', color=my_cmap[5])
        ax.plot(samples,values[:,6],label='SLS', color=my_cmap[6])
        ax.plot(samples,values[:,7],label='SRS', color=my_cmap[7])
    
        ax.set_title(title)
        ax.set_xlim(np.min(samples),np.max(samples))
    
    axes[0][0].set_ylabel("Path length")
    axes[1][0].set_ylabel("Path length")
    axes[0][0].legend()
    fig.tight_layout()
    fig.suptitle(f"Length fitting from {start} to {end}")
    plt.show()
    
    fig,ax = plt.subplots(figsize=(4,3))
    fig2,axes2 = plt.subplots(2,4)
    
    for i,(samples,values,title,marker,color) in enumerate(zip(list_samples,
                                                   list_data,
                                                   titles,
                                                   markers,
                                                   colors)):
        LRL_hist,_ = np.histogram(values[:,0],bins,target_lengths)
        RLR_hist,_ = np.histogram(values[:,1],bins,target_lengths)
        LSL_hist,_ = np.histogram(values[:,2],bins,target_lengths)
        LSR_hist,_ = np.histogram(values[:,3],bins,target_lengths)
        RSL_hist,_ = np.histogram(values[:,4],bins,target_lengths)
        RSR_hist,_ = np.histogram(values[:,5],bins,target_lengths)
        SLS_hist,_ = np.histogram(values[:,6],bins,target_lengths)
        SRS_hist,_ = np.histogram(values[:,7],bins,target_lengths)
        
        fig2.suptitle(title)
        axes2[0][0].bar(centers,LRL_hist,width=plot_width,color=my_cmap[0])
        axes2[0][0].set_title('LRL')
        axes2[0][1].bar(centers,RLR_hist,width=plot_width,color=my_cmap[1])
        axes2[0][1].set_title('RLR')
        axes2[0][2].bar(centers,LSL_hist,width=plot_width,color=my_cmap[2])
        axes2[0][2].set_title('LSL')
        axes2[0][3].bar(centers,LSR_hist,width=plot_width,color=my_cmap[3])
        axes2[0][3].set_title('LSR')
        axes2[1][0].bar(centers,RSL_hist,width=plot_width,color=my_cmap[4])
        axes2[1][0].set_title('RSL')
        axes2[1][1].bar(centers,RSR_hist,width=plot_width,color=my_cmap[5])
        axes2[1][1].set_title('RSR')
        axes2[1][2].bar(centers,SLS_hist,width=plot_width,color=my_cmap[6])
        axes2[1][2].set_title('SLS')
        axes2[1][3].bar(centers,SRS_hist,width=plot_width,color=my_cmap[7])
        axes2[1][3].set_title('SRS')
        plt.show()
        
        all_hists = [LRL_hist,RLR_hist,LSL_hist,LSR_hist,RSL_hist,RSR_hist,SLS_hist,SRS_hist]
        
        possibilites = sum(h > 0 for h in all_hists)
        
        ax.bar(centers+(i*plot_width/len(titles) - plot_width/2),possibilites,color=color,width=plot_width/len(titles),align='edge',label=title)
        
        color='0.1'
        ax.plot(centers,possibilites,color=color,
                marker=marker,markersize=7,
                linestyle=(0,(1,4)),
                label=title)
    
    ax.set_title(f"Number of length fitted paths for case $\\alpha={start_degrees}°$, $\\beta={end_degrees}°$, $d={d}$ using different methods with resolution {step}")
    ax.set_xlabel('Target length')
    ax.set_xlim(bins[0]-step,bins[-1]+step)
    ax.set_ylabel('Number of possible paths')
    ax.legend()
    fig.tight_layout()
    plt.show()

def plot_separation_sections(p1:Path, p2:Path, stats1:ACStats, stats2:ACStats, axe:Axes, colors:list[ColorType],
                             plot_distances:typing.Optional[typing.Literal['first','all','temporal','spatial']]=None):
    ## Compute full paths for legend
    dur1 = p1.total_length/stats1.airspeed
    dur2 = p2.total_length/stats2.airspeed
    dur = min(dur1,dur2)
    times = np.linspace(0,dur,100)
    pts1 = p1.poses_at(stats1.airspeed*times)
    pts2 = p2.poses_at(stats2.airspeed*times)
    _, l1s = plot_pose2d_sequence(axe,pts1,False,True,color='k',linestyle='solid',label='Path 1')
    _, l2s = plot_pose2d_sequence(axe,pts2,False,True,color='k',linestyle='dashed',label='Path 2')
    
    ## Compute sections
    lenghts1 = [s.length for s in p1.sections]
    lenghts2 = [s.length for s in p2.sections]
    cumdurations1 = np.cumsum(lenghts1)/stats1.airspeed
    cumdurations2 = np.cumsum(lenghts2)/stats2.airspeed
    sections_durations = sorted(list(set(cumdurations1).union(set(cumdurations2),[0])))
        
    ## Plot sections
    for i in range(1,len(sections_durations)):
        if sections_durations[i-1] > dur:
            break
        if abs(sections_durations[i] - sections_durations[i-1]) < 1e-6:
            continue
        times = np.linspace(sections_durations[i-1],min(sections_durations[i],dur),500)
        
        pts1 = p1.poses_at(stats1.airspeed*times)
        pts2 = p2.poses_at(stats2.airspeed*times)
        
        s1 = p1.section_at(stats1.airspeed*(sections_durations[i-1]+sections_durations[i])/2)
        s2 = p2.section_at(stats2.airspeed*(sections_durations[i-1]+sections_durations[i])/2)
        
        assert s1 is not None and s2 is not None
        
        char1 = 'S' if s1.type == DubinsMove.STRAIGHT else 'C'
        char2 = 'S' if s2.type == DubinsMove.STRAIGHT else 'C'
        
        plot_pose2d_sequence(axe,pts1,True,False,color=colors[i-1],linestyle='solid', label=f"Section {i}: {char1} -- {char2}")
        plot_pose2d_sequence(axe,pts2,True,False,color=colors[i-1],linestyle='dashed')
    
        if plot_distances is not None and (plot_distances != 'first' or (plot_distances == 'first' and i == 1)):
            ## Find minimum temporal distance between the two sections
            if plot_distances == 'temporal' or plot_distances == 'all':
                min_dist = np.inf
                min_dist_loc = 0
                
                for k,(pt1,pt2) in enumerate(zip(pts1,pts2)):
                    d = poses_XY_dist(pt1,pt2)
                    if d < min_dist:
                        min_dist = d
                        min_dist_loc = k
                        
                
            
                axe.plot([pts1[min_dist_loc].x,pts2[min_dist_loc].x],
                        [pts1[min_dist_loc].y,pts2[min_dist_loc].y],
                        marker='D',
                        color='k',
                        linestyle=':',markeredgecolor='r',
                        label=f"Temporal dist: {min_dist:.2f}")
            
            ## Find minimum spatial distance between the two sections
            if plot_distances == 'spatial' or plot_distances == 'all':
                min_dist = np.inf
                min_dist_loc = (0,0)
                for k1,pt1 in enumerate(pts1):
                    for k2,pt2 in enumerate(pts2):
                        d = poses_XY_dist(pt1,pt2)
                        if d < min_dist:
                            min_dist = d
                            min_dist_loc = (k1,k2)
                
                axe.plot([pts1[min_dist_loc[0]].x,pts2[min_dist_loc[1]].x],
                        [pts1[min_dist_loc[0]].y,pts2[min_dist_loc[1]].y],
                        marker='X',
                        color='k',
                        linestyle=':',markeredgecolor='r',
                        label=f"Spatial dist: {min_dist:.2f}")
                        
    
    axe.legend()
    axe.set_aspect('equal')
    
    ## Hide the initial full paths
    for l in l1s:
        l.set_visible(False)
    for l in l2s:
        l.set_visible(False)  
    plt.show()

def plot_pathplanning(stats:list[ACStats], starts:list[Pose2D], ends:list[Pose2D], target_length:float, separation:float):
    assert len(starts) == len(ends)
    
    fig,axe = plt.subplots(figsize=(16,9))
    
    paths_list:list[list[Path]] = []
    handles_list:list[list[Line2D]] = []
    
    # Plot one: show all possibilities
    
    for i,(s,e) in enumerate(zip(starts,ends)):
        c = my_cmap6[2*i % len(my_cmap6)]
        
        SLSL = plan_LSL_from_straights(stats[i],s,e,target_length,1.)
        LSLS = plan_LSL_from_straights(stats[i],s,e,target_length,0.)
        SRSR = plan_RSR_from_straights(stats[i],s,e,target_length,1.)
        RSRS = plan_RSR_from_straights(stats[i],s,e,target_length,0.)

        paths = [SLSL,LSLS,SRSR,RSRS]
        names = ["S-LSL","LSL-S","S-RSR","RSR-S"]
        markerstyles = ['+','+','o','o']
        marksersizes = [11,11,5,5]
        linestyles = ['--',':','--',':']
        
        paths_list.append(list(filter(lambda p : p is not None,paths))) # type: ignore
        handles = []

        for path,name,markerstyle,marksersize,linestyle in zip(paths,names,markerstyles,marksersizes,linestyles):
            if path is not None:
                l = path.total_length
                samples = np.linspace(0,l,100)
                points = []
                
                for i,s in enumerate(samples):
                    points.append(path.pose_at(samples[i]))
                
                
                _,lines = plot_pose2d_sequence(axe,points,False,True,label=name,
                    color=c, 
                    marker=markerstyle,markersize=marksersize,markevery=0.2,
                    linestyle=linestyle)
                handles.append(lines[0])
        
        handles_list.append(handles)
                
    
    for spine in axe.spines:
        axe.spines[spine].set_visible(False)
        
    axe.set_xticks([])
    axe.set_yticks([])
    axe.set_aspect('equal')
    axe.legend()
    fig.savefig("All_paths.svg")
    
    for handles in handles_list:
        for h in handles:
            h.set_alpha(0.2)
            
    for i in range(len(paths_list)):
        for j in range(i+1,len(paths_list)):
            print(f"Conflict checking between AC {stats[i].id} and AC {stats[j].id}")
            ps1     = paths_list[i]
            hdls1   = handles_list[i]
            
            ps2     = paths_list[j]
            hdls2   = handles_list[j]
            
            for t1,(p1,h1) in enumerate(zip(ps1,hdls1)):
                for t2,(p2,h2) in enumerate(zip(ps2,hdls2)):
                    
                    print(f"Checking between path {t1} of AC {stats[i].id} and path {t2} of AC {stats[j].id}")
                    
                    h1.set_alpha(1.)
                    h2.set_alpha(1.)
                    
                    dur1 = p1.total_length/stats[i].airspeed
                    dur2 = p2.total_length/stats[j].airspeed
                    
                    dur = min(dur1,dur2)
                    
                    times = np.linspace(0,dur,200)
                    
                    pts1 = p1.poses_at(stats[i].airspeed*times)
                    pts2 = p2.poses_at(stats[j].airspeed*times)
                    
                    min_dist = np.inf
                    min_dist_loc = 0
                    
                    for k,(pt1,pt2) in enumerate(zip(pts1,pts2)):
                        d = poses_XY_dist(pt1,pt2)
                        if d < min_dist:
                            min_dist = d
                            min_dist_loc = k
                    
                    out = axe.plot([pts1[min_dist_loc].x,pts2[min_dist_loc].x],
                                [pts1[min_dist_loc].y,pts2[min_dist_loc].y],
                                marker='D',
                                color='k' if min_dist > separation else 'r',
                                linestyle=':',markeredgecolor='r',
                                label=f"Min dist {min_dist:.2f} at t={times[min_dist_loc]:.2f}s")[0]
                    
                    axe.legend(handles=[h1,h2,out])
                    
                    fig.savefig(f"Conflict_AC{stats[i].id}_path{t1}_AC{stats[j].id}_path{t2}.svg")
                    h1.set_alpha(0.2)
                    h2.set_alpha(0.2)
                    out.remove()
        

if __name__ == '__main__':
    
    stats = ACStats(1,1.,0.,0.5)
    
    d = 4
    start_degrees   = 60
    end_degrees     = 330
    
    start   = Pose2D(0,0,np.deg2rad(start_degrees))
    end     = Pose2D(d,0,np.deg2rad(end_degrees))
    
    ## Plotting length variations
    
    # step = 0.5
    # plot_width = step*0.9
    
    # target_lengths = (0,15)
    # plot_length_functions(stats, d, start, end, step, plot_width, target_lengths)
    
    ## Plotting paths
    
    # plot_fits(stats,start,end)
    # plot_straights(stats, start, end, 10.)
    
    ## Demo path planning for two
    
    end2  = Pose2D(d-1.5,1,np.deg2rad(30))
    start2    = Pose2D(0.2,-2,np.deg2rad(-20))
    stats2  = ACStats(2,1.,0.,0.5)

    # plot_pathplanning([stats,stats2],[start,start2],[end,end2],10.,0.5)
    
    ## Demo separation sections
    
    p1_ref = plan_LSL(stats,start,end)
    p2_ref = plan_LSL(stats2,start2,end2)
    
    assert p1_ref is not None and p2_ref is not None
    
    target_l = max(p1_ref.total_length,p2_ref.total_length)*1.5
    
    r1 = fit_LSL_radius(start,end,stats.turn_radius,10*stats.turn_radius,target_l)
    r2 = fit_LSL_radius(start2,end2,stats2.turn_radius,10*stats2.turn_radius,target_l)
    
    assert r1 is not None and r2 is not None
    
    stats.turn_radius = r1
    stats2.turn_radius = r2
    
    p1 = plan_LSL(stats,start,end)
    p2 = plan_LSL(stats2,start2,end2)
    
    assert p1 is not None and p2 is not None
    assert abs(p1.total_length - target_l) < 1e-3 and abs(p2.total_length - target_l) < 1e-3
    
    # plot_separation_sections(p1,p2,stats,stats2,plt.subplots()[1],my_cmap6,'all')
    
    ## Demo full path interpolation
    
    # p1_interp = interpolate_path(stats,p1)
    # p2_interp = interpolate_path(stats2,p2)
    
    # fig,ax = plt.subplots()
    
    # samples1 = np.linspace(0,p1.total_length,100)
    
    # plot_pose2d_sequence(ax,p1.poses_at(samples1),False,True,color='blue',linestyle='dashed',label='Original path 1')
    # interp_pts1 = p1_interp(samples1)
    # ax.plot(interp_pts1[:,0],interp_pts1[:,1],color='blue',linestyle='solid',label='Interpolated path 1')
    
    # ax.set_aspect('equal')
    # ax.legend()
    # plt.show()
    
    ## Demo path section interpolation
    
    ref_poses = []
    interp_poses = []
    
    for s in p1.sections:
        samples = np.linspace(0,s.length,100)
        interp = unit_interpolate_basicpath(s,0.5)
        
        for l in samples:
            ref_poses.append(s.pose_at(l))
            interp_poses.append(interp(l))
            
    fig,ax = plt.subplots()
    
    interp_poses = np.asarray(interp_poses)
    print(interp_poses)
    
    plot_pose2d_sequence(ax,ref_poses,False,True,color='blue',linestyle='dashed',label='Original path 1')
    ax.plot(interp_poses[:,0],interp_poses[:,1],color='blue',linestyle='solid',label='Interpolated path 1')
    
    ax.set_aspect('equal')
    ax.legend()
    plt.show()
    
    