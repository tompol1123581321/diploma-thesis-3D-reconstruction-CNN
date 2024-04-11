import cv2
from matplotlib import pyplot as plt
import numpy as np

from calculate_camera_parameters.calculate_camera_parameters import calculate_camera_parameters
from depth_map_calculation.generate_depth_map import generate_depth_map
from disparity_calculation.disparity_map_generation.generate_disparity_map import generate_disparity_map
from g3D_mash_generation.generate_3D_mash import generate_mesh
from input_images.sdk.python.sintel_io import cam_read, disparity_read
from point_cloud_generation.generate_3D_point_cloud import depth_map_to_point_cloud
from result_visualisation.visualize_3D_mash import visualize_mesh
from result_visualisation.visualize_3D_point_cloud import visualize_3D_point_cloud


def load_and_normalize_disparity(image_path):
    # Load disparity image
    return disparity_read(image_path)


def visualize_disparity_map(disparity_map):
    plt.figure(figsize=(10, 7))
    plt.imshow(disparity_map, cmap='plasma')
    plt.colorbar()
    plt.title("Normalized Disparity Map")
    plt.show()

def preprocess_images(img1,img2):
    return img1,img2


def main():
    image_left_path = 'input_images/test_images/left.png'
    image_right_path = 'input_images/test_images/right.png'
    img1 = cv2.imread(image_left_path)
    img2 = cv2.imread(image_right_path)

    disparity_map = generate_disparity_map(img1,img2,"checkpoint_epoch_11.pth")

    focal_length, baseline = calculate_camera_parameters(disparity_map,10)

    depth_map = generate_depth_map(disparity_map,focal_length,baseline)
    point_cloud = depth_map_to_point_cloud(depth_map, focal_length)

    visualize_3D_point_cloud(point_cloud)

    mash_3D = generate_mesh(point_cloud) 
    visualize_mesh(mash_3D)


if __name__ == "__main__":
    main()
