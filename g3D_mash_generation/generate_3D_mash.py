import numpy as np
import open3d as o3d
import time

def generate_mesh(point_cloud, voxel_size=0.004, depth=9):
    print('Starting mesh generation...')
    
    # Check if the point cloud data is loaded correctly
    print(f'Point cloud data shape: {point_cloud.shape}')
    print('Sample points from the point cloud data:')
    print(point_cloud[:5])  # Print the first 5 points as a sample

    # Convert point cloud to Open3D format
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(point_cloud[:, 3:] / 255)  # Normalize color values to [0, 1]
    
    # Downsample the point cloud if a voxel size is given
    if voxel_size is not None:
        original_size = len(pcd.points)
        print(f'Original point cloud size: {original_size}')
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
        downsampled_size = len(pcd.points)
        print(f'Downsampling with voxel size {voxel_size}, new size: {downsampled_size}')
    
    # Estimate normals
    start_time = time.time()
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    normal_estimation_time = time.time() - start_time
    print(f"Normal estimation took {normal_estimation_time} seconds.")
    
    # Apply Poisson surface reconstruction
    print("Applying Poisson surface reconstruction...")
    start_time = time.time()
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth)
    mesh_generation_time = time.time() - start_time
    print(f"Mesh generation took {mesh_generation_time} seconds.")
    
    # Optionally: Remove low density vertices
    vertices_to_remove = densities < np.quantile(densities, 0.01)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    
    return mesh

