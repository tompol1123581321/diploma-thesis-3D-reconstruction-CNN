from numba import jit
import numpy as np
from scipy.optimize import differential_evolution

@jit(nopython=True)
def compute_depths(f, baseline, disparity_map):
    epsilon = 1e-6
    depths = (f * baseline) / (disparity_map + epsilon)
    return depths

@jit(nopython=True)
def compute_depth_variance(depths, disparity_map):
    valid_depths = []
    count = 0
    for i in range(depths.shape[0]):
        for j in range(depths.shape[1]):
            if disparity_map[i, j] > 0:
                valid_depths.append(depths[i, j])
                count += 1
    # Convert the list of valid depths to a numpy array
    valid_depths_array = np.array(valid_depths)
    depth_variance = np.var(valid_depths_array)
    return depth_variance

@jit(nopython=True)
def objective_function(x, disparity_map, baseline):
    f = x[0]
    depths = compute_depths(f, baseline, disparity_map)
    depth_variance = compute_depth_variance(depths, disparity_map)
    return depth_variance

def estimate_focal_length(disparity_map, baseline):
    bounds = [(500, 2000)]
    result = differential_evolution(objective_function, bounds, args=(disparity_map, baseline),
                                    strategy='best1bin', maxiter=100, popsize=15, tol=0.01, mutation=(0.5, 1.5), recombination=0.7)
    return result.x[0]
