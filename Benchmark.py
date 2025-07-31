# --- PyTorch Model Imports and Implementations ---
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import time
import csv
import argparse
from PIL import Image

# --- Custom Dataset Class ---
class CustomImageDataset(Dataset):
    """
    A custom dataset class to load images from a directory structure
    where each subdirectory represents a class.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the class subdirectories.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # Ensure class names are sorted for consistent indexing
        self.class_names = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        if not self.class_names:
            raise ValueError(f"No class subdirectories found in {root_dir}.")
            
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.class_names)}
        self.idx_to_class = {i: cls_name for cls_name, i in self.class_to_idx.items()}

        print(f"Found classes: {self.class_names}")

        # Populate image paths and labels
        for class_name in self.class_names:
            class_dir = os.path.join(root_dir, class_name)
            label_idx = self.class_to_idx[class_name]
            for img_name in os.listdir(class_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label_idx)

        if not self.image_paths:
            raise ValueError(f"No images found in any class subdirectories under {root_dir}.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        # Open image and convert to RGB to handle grayscale/multi-channel images consistently
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        # Return label as a long tensor for CrossEntropyLoss
        return image, torch.tensor(label, dtype=torch.long)

# LeNet is kept as a custom implementation
class LeNet(nn.Module):
    def __init__(self, num_classes=10, input_channels=3): # Default to 3 channels
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120) # Adjusted for 32x32 input
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# --- PyTorch Implementation of Conv-Node ---
class PyTorchConvNode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(PyTorchConvNode, self).__init__()
        # 'same' padding in PyTorch for Conv2d
        # For 'same' padding, the output spatial dimensions are the same as input
        # padding = kernel_size // 2 for odd kernel_size
        if padding == 'same':
            pad = kernel_size // 2
        else:
            pad = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.relu = nn.ReLU()

        # Residual connection projection if dimensions don't match
        self.shortcut_proj = None
        if in_channels != out_channels or stride != 1:
            self.shortcut_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.relu(out)

        if self.shortcut_proj is not None:
            identity = self.shortcut_proj(identity)
        
        # Ensure dimensions match for addition. If spatial dimensions differ due to stride,
        # the shortcut projection should have handled it.
        # If the number of channels still doesn't match after shortcut_proj (shouldn't happen if logic is correct),
        # or if out and identity have different spatial dimensions (e.g., if stride was applied to conv but not shortcut_proj correctly),
        # this might raise an error. The current shortcut_proj handles stride.
        
        # Pad identity if its spatial dimensions are smaller than out's due to 'valid' padding in conv
        # This is a common issue when converting from TF 'same' to PyTorch explicit padding.
        # However, with 'same' padding logic above, output spatial dims should match input.
        # If stride > 1, then output spatial dims will be reduced.
        # The shortcut_proj handles the stride, so spatial dims should align.

        # Check if spatial dimensions match before adding
        if out.shape[2:] != identity.shape[2:]:
            # This case should ideally be handled by the shortcut_proj's stride.
            # If it still occurs, it means the shortcut_proj didn't align dimensions.
            # For simplicity, we'll assume the shortcut_proj handles it.
            # If not, more complex padding/cropping might be needed here.
            pass # The shortcut_proj with stride handles spatial alignment.

        out = out + identity
        return out

# --- PyTorch Implementation of Multilayer Convolution Model ---
class PyTorchMultilayerConvolutionModel(nn.Module):
    def __init__(self, input_channels, num_classes, num_mlc_layers=2, num_conv_nodes_per_layer=3,
                 initial_filters=32, kernel_size=(3, 3), mlp_units=[128, 64]):
        super(PyTorchMultilayerConvolutionModel, self).__init__()
        
        self.num_mlc_layers = num_mlc_layers
        self.num_conv_nodes_per_layer = num_conv_nodes_per_layer
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size
        self.mlp_units = mlp_units
        self.output_classes = num_classes

        # Store MLC layers as a ModuleList
        self.mlc_layers = nn.ModuleList()
        current_in_channels = input_channels

        for i in range(num_mlc_layers):
            conv_nodes_in_layer = nn.ModuleList()
            filters = initial_filters * (2 ** i)
            for j in range(num_conv_nodes_per_layer):
                conv_nodes_in_layer.append(
                    PyTorchConvNode(current_in_channels, filters, kernel_size[0]) # Assuming square kernel
                )
            self.mlc_layers.append(conv_nodes_in_layer)
            # The output channels for the next layer will be the sum of filters
            # from all parallel conv nodes in the current layer.
            current_in_channels = filters * num_conv_nodes_per_layer

        # MLP layers
        self.flatten = nn.Flatten()
        
        mlp_layers = []
        # Calculate input features for the first MLP layer
        # This is tricky without knowing the exact spatial dimensions after convolutions.
        # A common approach is to pass a dummy tensor through the conv layers to get the shape.
        # For now, we'll assume the input_channels for the first MLP layer will be calculated dynamically in forward.
        
        # Dummy input for shape calculation (assuming 32x32 input for simplicity)
        # This will be refined in the forward pass for dynamic calculation.
        
        # The MLP layers will be defined in the forward pass or after a dummy run
        # to correctly determine the input features from the flattened output.
        # For now, we'll define them as a Sequential module once the input size is known.
        
        # Placeholder for MLP, actual layers will be created in forward
        self.mlp = None
        self.output_layer = None


    def forward(self, x):
        current_output = x

        for i, conv_nodes_in_layer in enumerate(self.mlc_layers):
            node_outputs = []
            for conv_node_module in conv_nodes_in_layer:
                node_outputs.append(conv_node_module(current_output))
            
            if len(node_outputs) > 1:
                current_output = torch.cat(node_outputs, dim=1) # Concatenate along channel dimension
            else:
                current_output = node_outputs[0]
        
        flatten_output = self.flatten(current_output)

        # Dynamically create MLP layers if not already created
        if self.mlp is None:
            mlp_input_features = flatten_output.shape[1]
            mlp_layers = []
            in_features = mlp_input_features
            for units in self.mlp_units:
                mlp_layers.append(nn.Linear(in_features, units))
                mlp_layers.append(nn.ReLU())
                in_features = units
            self.mlp = nn.Sequential(*mlp_layers).to(x.device) # Move to device

            self.output_layer = nn.Linear(in_features, self.output_classes).to(x.device) # Move to device
        
        mlp_output = self.mlp(flatten_output)
        final_output = self.output_layer(mlp_output)
        return final_output


def get_model(model_name, num_classes, input_channels=3, input_image_size=32):
    """
    Loads a model from torchvision or other public libraries,
    or returns the custom PyTorch MLC model.
    """
    print(f"Loading model: {model_name}")
    
    if model_name == 'LeNet':
        return LeNet(num_classes=num_classes, input_channels=input_channels)
    elif model_name == 'AlexNet':
        model = models.alexnet(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_name == 'VGG16':
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        return model
    elif model_name == 'ResNet18':
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'GoogLeNet':
        model = models.googlenet(weights=None, aux_logits=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    elif model_name == 'SqueezeNet':
        model = models.squeezenet1_0(weights=None)
        model.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1))
        model.num_classes = num_classes
        return model
    elif model_name == 'MobileNetV2':
        model = models.mobilenet_v2(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_name == 'DenseNet121':
        model = models.densenet121(weights=None)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
        return model
    elif model_name == 'EfficientNetB0':
        model = models.efficientnet_b0(weights=None)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_name == 'ViT':
        # ViT requires a specific image size (typically 224)
        if input_image_size != 224:
            print(f"Warning: ViT typically requires image_size=224. Current image_size={input_image_size}. "
                  "Adjusting transform or model might be needed for optimal performance.")
        model = models.vit_b_16(weights=None, image_size=input_image_size) # Pass image_size here
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
        return model
    elif model_name == 'GhostNet':
        # Requires: pip install timm
        try:
            model = torch.hub.load('huawei-noah/ghostnet', 'ghostnet_1x', pretrained=False)
            model.classifier = nn.Linear(model.classifier.in_features, num_classes)
            return model
        except Exception as e:
            print(f"Could not load GhostNet. Make sure 'timm' is installed. Error: {e}")
            raise
    elif model_name == 'VAN':
        # Requires: pip install transformers
        try:
            from transformers import VanForImageClassification
            model = VanForImageClassification.from_pretrained(
                'Visual-Attention-Network/van-base', num_labels=num_classes, ignore_mismatched_sizes=True)
            return model
        except Exception as e:
            print(f"Could not load VAN. Make sure 'transformers' is installed. Error: {e}")
            raise
    elif model_name == 'MLC':
        return PyTorchMultilayerConvolutionModel(
            input_channels=input_channels,
            num_classes=num_classes,
            num_mlc_layers=3, # Example: You can make these configurable if needed
            num_conv_nodes_per_layer=4,
            initial_filters=16,
            kernel_size=(3, 3),
            mlp_units=[64, 128]
        )
    else:
        raise NotImplementedError(f"Model '{model_name}' not implemented.")


# --- Trainer ---
class Trainer:
    def __init__(self, model, train_loader, val_loader, config, model_name):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=config.lr)
        self.log_file = f"{self.model_name}_log.csv"
        self.best_val_accuracy = 0.0

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0, 0, 0
        start_time = time.time()
        for data, targets in self.train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(data)
            # Check if the output is a tuple (like from GoogLeNet with aux_logits)
            # or if it has a 'logits' attribute (like from HuggingFace transformers)
            # Otherwise, assume it's directly the logits tensor.
            if isinstance(outputs, tuple):
                logits = outputs[0] # Take the main output
            elif hasattr(outputs, 'logits'):
                logits = outputs.logits
            else:
                logits = outputs
            
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            _, predicted = logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        return total_loss / len(self.train_loader), 100. * correct / total, time.time() - start_time

    def _evaluate(self, data_loader):
        self.model.eval()
        total_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                elif hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs
                loss = self.criterion(logits, targets)
                total_loss += loss.item()
                _, predicted = logits.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return total_loss / len(data_loader), 100. * correct / total

    def train(self):
        os.makedirs("saved_models", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        log_path = os.path.join("logs", self.log_file)
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'model_name', 'train_loss', 'train_accuracy', 
                             'val_loss', 'val_accuracy', 'epoch_time_seconds']) # Removed test_loss/accuracy as not used
        print(f"\n--- Training {self.model_name} on custom dataset ---")
        for epoch in range(self.config.epochs):
            train_loss, train_accuracy, epoch_time = self._train_epoch(epoch)
            val_loss, val_accuracy = self._evaluate(self.val_loader)
            print(f"Epoch {epoch+1}/{self.config.epochs} | "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}% | "
                  f"Time: {epoch_time:.2f}s")
            with open(log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch + 1, self.model_name, train_loss, train_accuracy,
                                 val_loss, val_accuracy, epoch_time])
            if val_accuracy > self.best_val_accuracy:
                self.best_val_accuracy = val_accuracy
                save_path = os.path.join("saved_models", f"{self.model_name}_best.pth")
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved best model to {save_path} with val acc: {val_accuracy:.2f}%")
        final_save_path = os.path.join("saved_models", f"{self.model_name}_final.pth")
        torch.save(self.model.state_dict(), final_save_path)
        print(f"Saved final model to {final_save_path}")

# --- Main Execution ---
def main():
    # --- Configuration ---
    # !! IMPORTANT !!
    # PLEASE SET THE PATH TO YOUR DATASET DIRECTORY HERE
    # This directory should contain subdirectories for each class (e.g., 'healthy', 'leukemia')
    DATA_DIR = "" 
    
    # --- Training Settings ---
    VAL_SPLIT = 0.2
    BATCH_SIZE = 32
    EPOCHS = 10
    LEARNING_RATE = 0.001

    # Create a simple config object to hold settings
    class Config:
        data_dir = DATA_DIR
        val_split = VAL_SPLIT
        batch_size = BATCH_SIZE
        epochs = EPOCHS
        lr = LEARNING_RATE
    config = Config()

    # --- Directory Check ---
    # Stop execution if the user has not changed the default path
    if not os.path.isdir(config.data_dir) or config.data_dir == "/path/to/your/dataset":
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: Please update the 'DATA_DIR' variable in the       !!!")
        print("!!! 'main' function to the correct path of your image dataset.!!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        return 

    # List of models to train. Now includes 'MLC' (PyTorch version)
    model_list = ['AlexNet','MLC', 'ViT', 'EfficientNetB0', 'GoogLeNet', 'ResNet18', 'AlexNet', 'VGG16', 'DenseNet121','SqueezeNet','MobileNetV2'] # Example: ['LeNet', 'ResNet18', 'MLC', 'ViT']

    for model_name in model_list:
        # Determine required image size for the current model
        if model_name in ['ViT', 'VAN', 'EfficientNetB0', 'GoogLeNet', 'ResNet18', 'AlexNet', 'VGG16', 'DenseNet121']:
            image_size = 224
        elif model_name == 'MLC':
            # MLC's initial layers are flexible, but a common starting size might be 32 or 64
            # Adjust this based on your expected input image size for the MLC model
            image_size = 32 
        else: # For LeNet, SqueezeNet, MobileNetV2
            image_size = 32
        
        # Define transforms based on the required image size
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load the full dataset
        full_dataset = CustomImageDataset(root_dir=config.data_dir, transform=transform)
        num_classes = len(full_dataset.class_names)
        
        # Split dataset into training and validation
        val_size = int(config.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

        # Get the model architecture
        # Pass input_channels to get_model for MLC
        model = get_model(model_name, num_classes, input_channels=3, input_image_size=image_size)

        # Print model parameter count
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model: {model_name}, Trainable Parameters: {total_params:,}")

        # Create a trainer and start training
        trainer = Trainer(model, train_loader, val_loader, config, model_name)
        trainer.train()

if __name__ == '__main__':
    main()
  
