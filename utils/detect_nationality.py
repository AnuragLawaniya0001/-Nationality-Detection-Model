# utils/detect_nationality.py

import torch
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

# Class labels (must match training folder structure order)
labels = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 'Middle Eastern', 'Southeast Asian', 'White']

# Load model
def load_nationality_model(model_path="models/nationality_7class_model.pth"):
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    model.fc = torch.nn.Linear(model.fc.in_features, len(labels))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model

# Predict
def predict_nationality(image_path, model):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(1).item()
    return labels[pred]
