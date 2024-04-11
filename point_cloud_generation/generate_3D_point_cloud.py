import cv2
import numpy as np
from open3d import geometry, utility, visualization

# Example usage:
cam_params = {
    'focal_length': 1000,  # Example focal length in pixels
    'baseline': 0.1,  # Example baseline in meters
    'principal_point': (320, 240),  # Example principal point (cx, cy)
    'extrinsic_matrix': np.eye(4)  # Example extrinsic matrix (identity matrix for simplicity)
}

import cv2
import numpy as np
from open3d import geometry, utility, visualization

def depth_map_to_point_cloud(depth_map, focal_length, optical_center):
    """
    Convert a depth map to a 3D point cloud.

    Parameters:
    - depth_map: A 2D numpy array containing depth values for each pixel.
    - focal_length: The focal length of the camera used to capture the depth map. It can be a tuple (fx, fy) for different focal lengths along the x and y axes or a single value if they are the same.
    - optical_center: The optical center (cx, cy) of the camera, typically the center of the image.

    Returns:
    - A 3D numpy array (N, 3) representing the point cloud, where N is the number of points. Each row contains the (x, y, z) coordinates of a point in 3D space.
    
    The function iterates over each pixel in the depth map, converting the pixel coordinates and depth value to a 3D point using the intrinsic camera parameters. This back-projection accounts for the camera's perspective projection model.
    """
    height, width = depth_map.shape
    if isinstance(focal_length, tuple):
        fx, fy = focal_length
    else:
        fx, fy = focal_length, focal_length
    cx, cy = optical_center
    
    # Create a meshgrid of pixel coordinates
    x = np.linspace(0, width - 1, width)
    y = np.linspace(0, height - 1, height)
    x, y = np.meshgrid(x, y)
    
    # Back-project to 3D space
    z = depth_map
    x = (x - cx) * z / fx
    y = (y - cy) * z / fy
    
    # Stack the coordinates
    points_3d = np.stack((x, y, z), axis=-1)
    
    # Reshape to a list of points
    points_3d = points_3d.reshape(-1, 3)
    
    # Remove points with zero depth
    points_3d = points_3d[~np.isinf(points_3d).any(axis=1)]
    
    return points_3d

