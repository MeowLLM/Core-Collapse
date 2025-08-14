import torch
from torchvision import models
import torch.nn as nn

num_classes = 6  # what you expect

# Initialize model
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

# Load weights
state_dict = torch.load("your_model_weights.pth", map_location=torch.device('cpu'))
model.load_state_dict(state_dict)

# --- Verification ---

# Check classifier output features
output_features = model.classifier[1].out_features
if output_features == num_classes:
    print(f"✅ Classifier matches expected number of classes: {output_features}")
else:
    print(f"❌ Classifier output ({output_features}) does NOT match expected {num_classes}")
