import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# --- 1. Initialize EfficientNet for 6 classes ---
num_classes = 6
model = models.efficientnet_b0(weights=None)  # no pretrained weights
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# --- 2. Load your trained weights ---
model.load_state_dict(torch.load("your_model_weights.pth", map_location=torch.device('cpu')))
model.eval()  # set to evaluation mode

# --- 3. Define image preprocessing ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --- 4. Load image ---
image_path = "path/to/your/image.jpg"
image = Image.open(image_path).convert("RGB")
input_tensor = preprocess(image).unsqueeze(0)  # add batch dimension

# --- 5. Run inference ---
with torch.no_grad():
    output = model(input_tensor)
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

# --- 6. Get predicted class ---
pred_class = torch.argmax(probabilities).item()
print("Predicted class index:", pred_class)
