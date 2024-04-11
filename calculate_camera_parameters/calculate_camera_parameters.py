from calculate_camera_parameters.estimate_focal_length import estimate_focal_length

def calculate_camera_parameters(disparity_map, baseline):

    estimated_focal_length = estimate_focal_length(disparity_map,baseline)

    return estimated_focal_length,baseline  


