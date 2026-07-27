from __future__ import annotations

import numpy as np
import typing
from dataclasses import dataclass,asdict

class Pose2D(typing.NamedTuple):
    x:float
    y:float
    angle:float
    
    def __str__(self) -> str:
        return f"$x={self.x:.3f},y={self.y:.3f},\\theta={np.rad2deg(self.angle):.2f}°$"

@dataclass
class Pose3D:
    x:float # X coordinate
    y:float # Y coordinate
    z:float # Z coordinate
    theta:float # XY orientation, in radian
    
    def __str__(self) -> str:
        return f"$x={self.x:.3f},y={self.y:.3f},z={self.z:.3f},\\theta={np.rad2deg(self.theta):.2f}°$"
    
    def to_numpy(self) -> np.ndarray:
        return np.array([self.x,self.y,self.z,self.theta])
    
    @staticmethod
    def from_array(array:typing.Sequence[float]|np.ndarray) -> Pose3D:
        return Pose3D(array[0],array[1],array[2],array[3])
    
    def asdict(self) -> dict[str,float]:
        return asdict(self)
    
    def astuple(self) -> tuple[float,float,float,float]:
        return (self.x,self.y,self.z,self.theta)
    
    @staticmethod
    def undefined() -> Pose3D:
        return Pose3D(np.nan,np.nan,np.nan,np.nan)
    
    def to2D(self) -> Pose2D:
        return Pose2D(self.x,self.y,self.theta)
        
def poses_dist(p1:Pose3D,p2:Pose3D) -> float:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    
    dtheta = p1.theta - p2.theta
    dvx = np.cos(dtheta) - 1.
    dvy = np.sin(dtheta) - 0
    
    return np.sqrt(dx*dx + dy*dy + dz*dz + dvx*dvx + dvy*dvy)

def poses_dist_2D(p1:Pose3D,p2:Pose3D) -> float:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    
    dtheta = p1.theta - p2.theta
    dvx = np.cos(dtheta) - 1.
    dvy = np.sin(dtheta) - 0
    
    return np.sqrt(dx*dx + dy*dy + dvx*dvx + dvy*dvy)

def poses_euclidean_dist(p1:Pose3D,p2:Pose3D) -> float:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    dz = p1.z - p2.z
    
    return np.sqrt(dx*dx + dy*dy + dz*dz)

def poses_XY_dist(p1:Pose3D|Pose2D,p2:Pose3D|Pose2D) -> float:
    dx = p1.x - p2.x
    dy = p1.y - p2.y
    return np.sqrt(dx*dx+dy*dy)

def min_XY_dist(poses:list[Pose3D]) -> tuple[float,int,int]:
    mdist = np.inf
    n = len(poses)
    i1 = 0
    i2 = 0
    
    for i in range(n):
        for j in range(i+1,n):
            dist = poses_XY_dist(poses[i],poses[j])
            if dist < mdist:
                mdist = dist
                i1 = i
                i2 = j
                
    return mdist,i1,i2

TimedPosesLine = tuple[float,dict[int,Pose3D]]
ListOfTimedPoses = list[TimedPosesLine]
DictOfPoseTrajectories = dict[int,list[tuple[float,Pose3D]]]

