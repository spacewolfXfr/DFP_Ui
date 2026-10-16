from DubinsFleetPlanner.Poses import Pose3D,ListOfTimedPoses
from DubinsFleetPlanner.Dubins import ACStats,Path

def insert_delayed_poses(poses:ListOfTimedPoses,stat:ACStats,path:Path,delay:float) -> ListOfTimedPoses:
    """ Insert a delayed path into a list of already sampled poses.
    We require the list of poses to be sorted by increasing time stamps.

    Args:
        poses (ListOfTimedPoses): Reference list of poses
        stat (ACStats): Delayed path aircraft statistics
        path (Path): Delayed path description
        delay (float): Delay

    Raises:
        ValueError: List of poses is empty

    Returns:
        ListOfTimedPoses: Input list with same time stamps but added poses from the delayed path
    """
    if len(poses) == 0:
        raise ValueError("Poses list is empty")
    
    for t,d in poses:
        if t < delay:
            d[stat.id] = Pose3D.undefined()
        else:
            pose = path.pose_at_length((t-delay)*stat.airspeed)
            d[stat.id] = pose
    
    return poses