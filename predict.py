import torch
from torchvision import transforms
from PIL import Image
import os
import sys
from new_model import CNNModel

# Define the transform to preprocess the image
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

def load_model(model_path):
    model = CNNModel()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        model.eval()  # Set the model to evaluation mode
    else:
        raise FileNotFoundError(f'Model file {model_path} does not exist.')
    return model

def predict_image(image_path, model_path):
    model = load_model(model_path)
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = model(image)
    return output.item()  # Convert tensor to a single value

if __name__ == "__main__":
    model_path = "modelv5b1.pth"
    print(predict_image('images/0a0a9d8dd3cd457e9aa04316ca5e8322.jpg', model_path))