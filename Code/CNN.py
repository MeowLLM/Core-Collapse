import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.metrics import accuracy_score

# Config
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.00005
DATA_DIR = ""

best_acc = 0.0
best_model_path = "best_model_alexnet.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# Split train/val
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# Model setup
model = models.alexnet(weights=None)
model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"[Epoch {epoch+1}/{EPOCHS}] Training Loss: {total_loss / len(train_loader):.4f}")

    # Validation
    print("\nStarting validation...")
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)  # FIXED: no `.logits`
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    real_acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Real Accuracy (overall): {real_acc:.4f}")

    # Save best model
    if real_acc > best_acc:
        best_acc = real_acc
        torch.save(model.state_dict(), best_model_path)
        print(f"📦 Saved new best model with accuracy {best_acc:.4f} to {best_model_path}")
