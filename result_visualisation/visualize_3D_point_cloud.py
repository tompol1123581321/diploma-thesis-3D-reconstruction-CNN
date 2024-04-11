import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualize_3D_point_cloud(point_cloud):
    view_angle = 180
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    # Extract the 3D points and colors
    points_3D = point_cloud[:, :3]
    colors = point_cloud[:, 3:] / 255  # Normalize color values to [0, 1] for matplotlib
    
    # Compute the centroid of the point cloud
    centroid = np.mean(points_3D, axis=0)
    
    # Rotate the point cloud
    R = np.array([[np.cos(np.radians(view_angle)), -np.sin(np.radians(view_angle)), 0],
                  [np.sin(np.radians(view_angle)), np.cos(np.radians(view_angle)), 0],
                  [0, 0, 1]])
    rotated_points_3D = np.dot(points_3D - centroid, R)

    # Plotting the rotated point cloud
    ax.scatter(rotated_points_3D[:, 0], rotated_points_3D[:, 1], rotated_points_3D[:, 2], c='blue')


    # Set the angle of the camera
    ax.view_init(elev=10., azim=view_angle)

    # Set axis labels
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')

    plt.show()
