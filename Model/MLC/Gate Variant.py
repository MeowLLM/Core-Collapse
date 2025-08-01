import torch
import torch.nn as nn

class PyTorchConvNode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(PyTorchConvNode, self).__init__()
        
        if padding == 'same':
            pad = kernel_size // 2
        else:
            pad = 0

        # Main convolutional path
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.relu = nn.ReLU()
        
        # --- Correct Gating Mechanism ---
        # The gate is a SEPARATE convolution that learns to produce the gate values.
        # It has the same number of output channels as the main convolution.
        self.gate_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        self.sigmoid = nn.Sigmoid()
        
        # Shortcut connection
        self.shortcut_proj = None
        if in_channels != out_channels or stride != 1:
            self.shortcut_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x

        # Get the gate values from the separate gate_conv path
        gate_values = self.sigmoid(self.gate_conv(x))
    
        # Pass the original input through the main convolutional path
        main_path_output = self.relu(self.conv(x))
        
        # Use the gate values to multiply the main path's output
        # This is where the flow is controlled.
        gated_output = main_path_output * gate_values

        # Handle the residual shortcut connection
        if self.shortcut_proj is not None:
            identity = self.shortcut_proj(identity)
            
        # Add the gated convolutional output to the identity shortcut
        out = gated_output + identity
        
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
