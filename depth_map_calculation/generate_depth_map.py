import numpy as np

def generate_depth_map(disparity_map, assumed_focal_length=700, assumed_baseline=0.1):
    print("Converting disparity map to depth map...")
    
    # Avoid division by zero by temporarily setting zero disparities to 'inf',
    # which results in zero depth, indicating immeasurable or infinite distance.
    valid_disparity = np.where(disparity_map > 0, disparity_map, np.inf)
    
    # Convert disparity to depth
    depth_map = (assumed_focal_length * assumed_baseline) / valid_disparity
    
    return depth_map