import torch
from torch import nn
import torch.nn.functional as F

class PyTorchGatedConvNode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(PyTorchGatedConvNode, self).__init__()
        
        if padding == 'same':
            pad = kernel_size // 2
        else:
            pad = 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.relu = nn.ReLU()
        
        # Correct Gating Mechanism: Separate convolution for gate values
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.sigmoid = nn.Sigmoid()
        
        self.shortcut_proj = None
        if in_channels != out_channels or stride != 1:
            self.shortcut_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x
        gate_values = self.sigmoid(self.gate_conv(x))
        main_path_output = self.relu(self.conv(x))
        gated_output = main_path_output * gate_values
        if self.shortcut_proj is not None:
            identity = self.shortcut_proj(identity)
        out = gated_output + identity
        return out


# --- A Gated Classifier Head ---
class GatedClassifierHead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # The gate layer learns to create gate values based on the hidden features
        self.gate_layer = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # 1. Create the main data path features
        h1 = F.relu(self.fc1(x))
        
        # 2. Create the gate values from a separate path
        gate_val = torch.sigmoid(self.gate_layer(x))

        # 3. Apply the gate to the main data path
        h1_gated = h1 * gate_val
        
        # 4. Final classification layer
        return self.fc2(h1_gated)


# --- PyTorch Multilayer Convolution Model with Gated Classifier Head ---
class PyTorchMultilayerConvolutionModelWithGatedHead(nn.Module):
    def __init__(self, input_channels, num_classes, num_mlc_layers=2, num_conv_nodes_per_layer=3,
                 initial_filters=32, kernel_size=(3, 3), mlp_units=[128]):
        super().__init__()
        
        self.num_mlc_layers = num_mlc_layers
        self.num_conv_nodes_per_layer = num_conv_nodes_per_layer
        self.initial_filters = initial_filters
        self.kernel_size = kernel_size
        self.mlp_units = mlp_units
        self.output_classes = num_classes

        self.mlc_layers = nn.ModuleList()
        current_in_channels = input_channels

        for i in range(num_mlc_layers):
            conv_nodes_in_layer = nn.ModuleList()
            filters = initial_filters * (2 ** i)
            for j in range(num_conv_nodes_per_layer):
                # Using the corrected, gated conv node
                conv_nodes_in_layer.append(
                    PyTorchGatedConvNode(current_in_channels, filters, kernel_size[0])
                )
            self.mlc_layers.append(conv_nodes_in_layer)
            current_in_channels = filters * num_conv_nodes_per_layer

        self.flatten = nn.Flatten()
        
        # Placeholder for the gated classifier, will be created dynamically
        self.gated_classifier_head = None


    def forward(self, x):
        current_output = x

        for i, conv_nodes_in_layer in enumerate(self.mlc_layers):
            node_outputs = []
            for conv_node_module in conv_nodes_in_layer:
                node_outputs.append(conv_node_module(current_output))
            
            if len(node_outputs) > 1:
                current_output = torch.cat(node_outputs, dim=1)
            else:
                current_output = node_outputs[0]
        
        flatten_output = self.flatten(current_output)

        # Dynamically create the gated classifier head if not already created
        if self.gated_classifier_head is None:
            mlp_input_features = flatten_output.shape[1]
            hidden_dim = self.mlp_units[0] if self.mlp_units else 128
            self.gated_classifier_head = GatedClassifierHead(
                input_dim=mlp_input_features,
                hidden_dim=hidden_dim,
                output_dim=self.output_classes
            ).to(x.device)
        
        final_output = self.gated_classifier_head(flatten_output)
        return final_output
