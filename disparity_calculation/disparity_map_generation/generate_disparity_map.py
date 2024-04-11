import torch
import cv2
import numpy as np
from disparity_calculation.feature_matching_CNN_model.custom_cnn_model import EnhancedCustomCNN

def load_pretrained_model(model_path, disparity_shape):
    model = EnhancedCustomCNN(disparity_shape)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully.")
    return model

def preprocess_image(image, target_height, target_width):
    aspect_ratio = image.shape[1] / image.shape[0]
    new_width = int(target_height * aspect_ratio)
    resized_image = cv2.resize(image, (new_width, target_height), interpolation=cv2.INTER_AREA)
    delta_w = target_width - new_width
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    color = [0, 0, 0]  # Black padding
    padded_image = cv2.copyMakeBorder(resized_image, 0, 0, left, right, cv2.BORDER_CONSTANT, value=color)
    print(f"Original image shape: {image.shape}")
    print(f"Resized image shape: {resized_image.shape}")
    print(f"Padded image shape: {padded_image.shape}")
    return padded_image

def postprocess_disparity_map(disparity_map):
    print(f"Raw disparity map shape: {disparity_map.shape}")
    # Remove all singleton dimensions
    disparity_map = disparity_map.squeeze()
    print(f"Post-processed disparity map shape: {disparity_map.shape}")
    return disparity_map

def can_save_disparity_map(disparity_map):
    if not isinstance(disparity_map, np.ndarray):
        print("Disparity map is not a numpy array.")
        return False
    if disparity_map.ndim != 2:
        print("Disparity map does not have 2 dimensions.")
        return False
    if disparity_map.dtype not in [np.uint8, np.uint16, np.float32]:
        print(f"Disparity map data type is {disparity_map.dtype}, which is not supported by OpenCV.")
        return False
    if (disparity_map < 0).any():
        print("Disparity map contains negative values.")
        return False
    return True

def save_disparity_map(disparity_map, filename):
    if not can_save_disparity_map(disparity_map):
        print("Cannot save disparity map due to invalid format or data type.")
        return
    cv2.imwrite(filename, disparity_map)
    print(f"Disparity map saved to {filename}")

def generate_disparity_map(left_image, right_image, model_path, expected_size=(436, 1024)):
    model = load_pretrained_model(model_path, expected_size)
    left_img = preprocess_image(left_image, expected_size[0], expected_size[1])
    right_img = preprocess_image(right_image, expected_size[0], expected_size[1])

    left_img_tensor = torch.from_numpy(left_img).float().permute(2, 0, 1).unsqueeze(0)  # Add batch dimension
    right_img_tensor = torch.from_numpy(right_img).float().permute(2, 0, 1).unsqueeze(0)  # Add batch dimension

    if torch.cuda.is_available():
        left_img_tensor = left_img_tensor.cuda()
        right_img_tensor = right_img_tensor.cuda()

    with torch.no_grad():
        disparity_map_tensor = model(left_img_tensor, right_img_tensor)
        disparity_map = disparity_map_tensor.squeeze().cpu().numpy()

    processed_disparity_map = postprocess_disparity_map(disparity_map)
    print(f"Disparity map value range before scaling: min = {processed_disparity_map.min()}, max = {processed_disparity_map.max()}")

    if processed_disparity_map.max() <= 1.0:
        print("Scaling disparity map by 255.")
        processed_disparity_map = np.clip(processed_disparity_map, 0, 1)
        processed_disparity_map = (processed_disparity_map * 255).astype(np.uint8)

    save_disparity_map(processed_disparity_map, "results/current_disparity_map.jpg")
    return processed_disparity_map