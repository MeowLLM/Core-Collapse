import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, models, datasets
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# ===== CONFIG =====
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.00005
DATA_DIR = "D:\\Leukemia"  # <<< Dataset path

best_acc = 0.0
best_model_path = "best_model_efficientnet.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ===== Data transforms =====
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ===== Load dataset =====
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
class_names = dataset.classes
num_classes = len(class_names)
print(f"Found {num_classes} classes: {class_names}")

# ===== Split train/val =====
val_size = int(0.2 * len(dataset))
train_size = len(dataset) - val_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===== Model setup =====
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# ===== Metric storage =====
acc_list, precision_list, recall_list, f1_list = [], [], [], []

# ===== Training loop =====
print("\nStarting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # ===== Validation =====
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1).cpu().numpy()
            y_pred.extend(preds)
            y_true.extend(labels.numpy())

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    acc_list.append(acc)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)

    print(f"[Epoch {epoch+1}/{EPOCHS}] Loss: {avg_loss:.4f} | Acc: {acc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")

    # Save best model
    if acc > best_acc:
        best_acc = acc
        best_y_true = y_true  # store last true labels for best model
        best_y_pred = y_pred  # store last pred labels for best model
        torch.save(model.state_dict(), best_model_path)

# ===== Show averages =====
print("\n===== Average Metrics Across Epochs =====")
print(f"Avg Accuracy: {np.mean(acc_list):.4f}")
print(f"Avg Precision: {np.mean(precision_list):.4f}")
print(f"Avg Recall: {np.mean(recall_list):.4f}")
print(f"Avg F1-score: {np.mean(f1_list):.4f}")

# ===== Final confusion matrix (from best epoch) =====
cm = confusion_matrix(best_y_true, best_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (Best Model)")
plt.show()

# ===== TP, FP, FN, TN =====
TP = np.diag(cm)
FP = cm.sum(axis=0) - TP
FN = cm.sum(axis=1) - TP
TN = cm.sum() - (FP + FN + TP)

print("\nPer-class metrics:")
for idx, cls in enumerate(class_names):
    print(f"Class: {cls}")
    print(f"   TP: {TP[idx]}")
    print(f"   FP: {FP[idx]}")
    print(f"   FN: {FN[idx]}")
    print(f"   TN: {TN[idx]}")
    print("------")

print("\nOverall:")
print(f"TP: {TP.sum()}, FP: {FP.sum()}, FN: {FN.sum()}, TN: {TN.sum()}")

# ===== Plot TP, FP, FN, TN per class =====
metrics_data = {
    'TP': TP,
    'FP': FP,
    'FN': FN,
    'TN': TN
}
metrics_labels = list(metrics_data.keys())

plt.figure(figsize=(10, 6))
bar_width = 0.2
x = np.arange(len(class_names))

for i, metric in enumerate(metrics_labels):
    plt.bar(x + i*bar_width, metrics_data[metric], width=bar_width, label=metric)

plt.xticks(x + bar_width * 1.5, class_names)
plt.ylabel("Count")
plt.title("TP, FP, FN, TN per Class (Best Model)")
plt.legend()
plt.show()
