import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import transforms
from PIL import Image

# 1. Re‑create the model architecture
NUM_CLASSES = 7
device = torch.device("cpu")  # or "cuda" if available

model = resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(device)

# 2. Load the saved weights
state_dict = torch.load("models/nationality_7class_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()

# 3. Class labels (alphabetical order of your folders)
labels = ['Black', 'East Asian', 'Indian', 'Latino_Hispanic', 
          'Middle Eastern', 'Southeast Asian', 'White']

# 4. Prediction function
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def predict_nationality(image_path: str) -> str:
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        idx = out.argmax(dim=1).item()
    return labels[idx]

# 5. Quick test
if __name__ == "__main__":
    test_img = fr"c:\Users\Anurag Lawaniya\Pictures\123.jpg"
    print("Predicted nationality:", predict_nationality(test_img))
