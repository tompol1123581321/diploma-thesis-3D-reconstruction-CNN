import os
import cv2
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset
import torch



class StereoDataset(Dataset):
    def __init__(self, left_images, right_images, disparities, augment=False):
        self.left_images = left_images
        self.right_images = right_images
        self.disparities = disparities
        self.disparity_shape = disparities[0].shape
        self.augment = augment

        # Data augmentation
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),  # rotations up to 10 degrees
            # Add more transformations as needed
        ])

    def __len__(self):
        return len(self.left_images)

    def __getitem__(self, idx):
        left_img = torch.from_numpy(self.left_images[idx]).permute(2, 0, 1).float() / 255.0
        right_img = torch.from_numpy(self.right_images[idx]).permute(2, 0, 1).float() / 255.0
        disparity = torch.from_numpy(self.disparities[idx]).unsqueeze(0).float()

        # Apply transformations if augmentation is True
        if self.augment:
            left_img = self.transform(left_img)
            right_img = self.transform(right_img)
            disparity = self.transform(disparity)

        # Resize disparity to match the output size
        disparity = torch.nn.functional.interpolate(disparity.unsqueeze(0), size=self.disparity_shape, mode='nearest')
        disparity = disparity.squeeze(0)  # Remove the batch dimension
        return left_img, right_img, disparity

# Load and split the data
def load_data(data_dir, test_size=0.2):
    left_images = []
    right_images = []
    disparities = []

    for scene_name in os.listdir(os.path.join(data_dir, "clean_left")):
        for image_name in os.listdir(os.path.join(data_dir, "clean_left", scene_name)):
            left_images.append(cv2.imread(os.path.join(data_dir, 'clean_left', scene_name, image_name)))
            right_images.append(cv2.imread(os.path.join(data_dir, 'clean_right', scene_name, image_name)))
            disparities.append(cv2.imread(os.path.join(data_dir, 'disparities', scene_name, image_name), cv2.IMREAD_GRAYSCALE))

    train_left, val_left, train_right, val_right, train_disp, val_disp = train_test_split(
        left_images, right_images, disparities, test_size=test_size, random_state=42)

    return train_left, train_right, train_disp, val_left, val_right, val_disp