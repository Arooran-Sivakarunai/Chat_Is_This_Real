import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os

# Define the dataset
class ImageDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 1]  # Access the file path
        image = Image.open(img_name).convert('RGB')  # Convert image to RGB
        label = self.data_frame.iloc[idx, 2]  # Access the label

        if self.transform:
            image = self.transform(image)

        return image, label

# Define the CNN model
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 30 * 30, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        # x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.dropout(x)
        # x = self.pool(F.relu(self.conv3(x)))
        # x = self.dropout(x)
        x = x.view(-1, 64 * 30 * 30)
        x = F.relu(self.fc1(x))
        # x = self.dropout(x)
        x = self.sigmoid(self.fc2(x))
        return x

if __name__ == "__main__":
    # Save the initial model
    model_save_path = 'modelv2b1.pth'
    model = CNNModel()
    torch.save(model.state_dict(), model_save_path)
    print("Initial model saved successfully")