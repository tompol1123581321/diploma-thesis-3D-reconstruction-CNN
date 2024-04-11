import os
import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

from custom_cnn_model import EnhancedCustomCNN
from data_preperation import StereoDataset, load_data

def get_disparity_shape(disparities_dir):
    try:
        sample_disparity = cv2.imread(os.path.join(disparities_dir, os.listdir(disparities_dir)[0]), cv2.IMREAD_GRAYSCALE)
        if sample_disparity is not None:
            return sample_disparity.shape
        else:
            raise ValueError("Disparity map is None.")
    except Exception as e:
        print("Error loading disparity map:", e)
        return None

def train_custom_cnn(data_dir, num_epochs=12, patience=5):
    disparity_shape = get_disparity_shape(os.path.join(data_dir, 'disparities', os.listdir(os.path.join(data_dir, 'disparities'))[0]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = EnhancedCustomCNN(disparity_shape).to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    train_left, train_right, train_disp, val_left, val_right, val_disp = load_data(data_dir)
    train_dataset = StereoDataset(train_left, train_right, train_disp, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
    val_dataset = StereoDataset(val_left, val_right, val_disp)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=4)

    scaler = GradScaler()
    
    best_val_loss = float('inf')
    epochs_since_improvement = 0
    save_checkpoint_frequency = 2

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        print("training started")
        for left_img, right_img, disparity in train_loader:
            left_img, right_img, disparity = left_img.to(device), right_img.to(device), disparity.to(device)
            print("training image")

            optimizer.zero_grad()
            with autocast():
                outputs = model(left_img, right_img)
                loss = criterion(outputs, disparity)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * left_img.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss}")
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for left_img, right_img, disparity in val_loader:
                left_img, right_img, disparity = left_img.to(device), right_img.to(device), disparity.to(device)
                outputs = model(left_img, right_img)
                val_loss += criterion(outputs, disparity).item()
        
        val_loss /= len(val_loader.dataset)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")
        scheduler.step()
        
        # Checkpointing
        if epoch % save_checkpoint_frequency == 0 or epoch == num_epochs - 1:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch}.pth")
        
        # Early Stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_since_improvement = 0
            # Save the best model
            torch.save(model.state_dict(), "best_cnn_disparity_map_generator.pth")
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print('Early stopping triggered!')
                break

# Uncomment the next line to train the model when this script is run
train_custom_cnn("input_images/training")
