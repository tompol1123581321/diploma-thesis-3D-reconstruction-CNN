import open3d as o3d


def visualize_mesh(mesh):
    # Visualize the mesh
    o3d.visualization.draw_geometries([mesh], window_name="3D Mesh")