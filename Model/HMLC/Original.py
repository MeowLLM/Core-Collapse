import torch
import torch.nn as nn
import torch.nn.functional as F
import re

class EReLU(nn.Module):
    """
    Custom EReLU activation function.
    This activation function has learnable parameters to adapt its shape.
    """
    def __init__(
        self, 
        init_a=0.0, 
        init_b=1.0, 
        init_c=0.5, 
        init_d=0.0, 
        init_i=1.0, 
        eps=1e-6, 
        safety_mode="close"  # options: "too_close", "close"
    ):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(init_a))
        self.b = nn.Parameter(torch.tensor(init_b))
        self.c = nn.Parameter(torch.tensor(init_c))
        self.d = nn.Parameter(torch.tensor(init_d))
        self.i = nn.Parameter(torch.tensor(init_i))
        self.eps = eps
        self.safety_mode = safety_mode

    def forward(self, x):
        a, b, c, d, i = self.a.to(x), self.b.to(x), self.c.to(x), self.d.to(x), self.i.to(x)
        numerator = a + b * x - c * torch.abs(x) + d

        if self.safety_mode == "too_close":
            # only replace if |i| is almost zero
            safe_i = torch.where(torch.abs(i) < self.eps,
                                 torch.tensor(torch.e, device=x.device, dtype=x.dtype),
                                 i)
        elif self.safety_mode == "close":
            # smoothly push i away from zero: add eps
            safe_i = i + self.eps * torch.sign(i)  # nudges i away from 0
        else:
            raise ValueError(f"Unknown safety_mode: {self.safety_mode}")

        return numerator / safe_i

class PyTorchConvNode(nn.Module):
    """
    Custom Convolutional Node with Gated Connections and EReLU activation.
    This is the core building block for our network branches.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(PyTorchConvNode, self).__init__()
        
        if padding == 'same':
            pad = kernel_size // 2
        else:
            pad = 0

        # Gating mechanisms for the main path
        self.gate_input = nn.Hardsigmoid()
        self.gate_output = nn.Hardsigmoid()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad)
        # Using the custom EReLU activation function instead of standard ReLU
        self.relu = EReLU()
        
        # Residual connection projection if dimensions don't match
        self.shortcut_proj = None
        if in_channels != out_channels or stride != 1:
            self.shortcut_proj = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        identity = x

        # Apply the "Gate Input" to the main path's tensor
        gated_input = self.gate_input(x)
        
        # Pass the gated input through the convolution and EReLU
        out = self.conv(gated_input)
        out = self.relu(out)
        
        # Apply the "Gate Output" to the result of the main path
        out = self.gate_output(out)

        # Handle the residual shortcut connection
        if self.shortcut_proj is not None:
            identity = self.shortcut_proj(identity)
            
        # Add the gated convolutional output to the non-gated identity shortcut
        out = out + identity
        return out

class ConfigurableHighwayNet(nn.Module):
    """
    A highly configurable neural network that dynamically builds parallel branches
    based on user-defined specifications.
    """
    def __init__(self, in_channels=3, num_classes=10, branch_configs=None, 
                 initial_channels=64, mlp_hidden_dim=512):
        super(ConfigurableHighwayNet, self).__init__()

        if branch_configs is None:
            # Provide a default configuration if none is given
            branch_configs = [
                {'source_layer': 0, 'refinements': 3, 'pooling': 0},
                {'source_layer': 0, 'refinements': 2, 'pooling': 2},
                {'source_layer': 0, 'refinements': 1, 'pooling': 4},
            ]
        
        self.branch_configs = branch_configs
        self.num_classes = num_classes
        self.mlp_hidden_dim = mlp_hidden_dim
        
        # --- Main Stem (Initial Feature Extractor) ---
        # This is the initial layer that all branches will sprout from.
        self.stem = PyTorchConvNode(in_channels, initial_channels, kernel_size=3)

        # --- Dynamic Branch and MLP Creation ---
        self.branches = nn.ModuleDict()
        self.mlps = nn.ModuleDict()
        self._mlps_initialized = False

        for i, config in enumerate(self.branch_configs):
            branch_layers = []
            
            # 1. Add Pooling Layer if specified
            pooling_size = config.get('pooling', 0)
            if isinstance(pooling_size, str): # e.g., '32x32'
                dims = [int(d) for d in re.findall(r'\d+', pooling_size)]
                if len(dims) == 2:
                    branch_layers.append(nn.AdaptiveAvgPool2d(tuple(dims)))
            elif isinstance(pooling_size, int) and pooling_size > 1:
                branch_layers.append(nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_size))

            # 2. Add Refinement Layers (PyTorchConvNode)
            refinements = config.get('refinements', 1)
            # Channel sizes will double for each pooling level for complexity
            current_channels = initial_channels * (2**i) if i > 0 else initial_channels
            
            # The first conv node transitions from initial_channels to the branch's channel size
            branch_layers.append(PyTorchConvNode(initial_channels, current_channels, kernel_size=3))

            for _ in range(refinements - 1): # -1 because we already added one
                branch_layers.append(PyTorchConvNode(current_channels, current_channels, kernel_size=3))
            
            self.branches[f'branch_{i}'] = nn.Sequential(*branch_layers)

    def _initialize_mlps(self, x):
        """
        Dynamically calculates flattened feature sizes and initializes MLP layers
        after the first forward pass.
        """
        print("--- Initializing MLP layers dynamically ---")
        stem_features = self.stem(x)
        
        total_mlp_inputs = 0
        
        for i, config in enumerate(self.branch_configs):
            branch_out = self.branches[f'branch_{i}'](stem_features)
            flat_size = branch_out.numel() // branch_out.shape[0]
            print(f"  Branch {i} flattened size: {flat_size}")
            
            self.mlps[f'mlp_{i}'] = nn.Sequential(
                nn.Linear(flat_size, self.mlp_hidden_dim),
                EReLU(),
                nn.Dropout(0.5)
            )
            total_mlp_inputs += self.mlp_hidden_dim

        # Final ensemble layer
        self.ensemble_mlp = nn.Linear(total_mlp_inputs, self.num_classes)
        self._mlps_initialized = True
        print("--- MLP initialization complete ---")


    def forward(self, x):
        """Defines the forward pass of the model."""
        if not self._mlps_initialized:
            # Move model to the same device as input before initialization
            self.to(x.device)
            self._initialize_mlps(x)

        # 1. Pass input through the main stem
        stem_features = self.stem(x)
        
        # 2. Process features through each parallel branch
        mlp_outputs = []
        for i, config in enumerate(self.branch_configs):
            branch_out = self.branches[f'branch_{i}'](stem_features)
            branch_flat = torch.flatten(branch_out, 1)
            mlp_out = self.mlps[f'mlp_{i}'](branch_flat)
            mlp_outputs.append(mlp_out)
            
        # 3. Ensemble the results
        ensemble_in = torch.cat(mlp_outputs, dim=1)
        final_output = self.ensemble_mlp(ensemble_in)
        
        return final_output

# --- Example Usage ---
if __name__ == '__main__':
    # Define a custom configuration for the network's branches
    # Each dictionary defines one parallel branch.
    # 'source_layer' is currently 0 for all, meaning they all branch from the same stem.
    # 'refinements' is the number of ConvNodes in the branch.
    # 'pooling' can be 0 (no pooling), an integer (MaxPool), or a string 'HxW' (AdaptivePool).
    custom_config = [
        {'source_layer': 0, 'refinements': 4, 'pooling': 0},          # High-res branch
        {'source_layer': 0, 'refinements': 3, 'pooling': 2},          # Mid-res branch
        {'source_layer': 0, 'refinements': 2, 'pooling': '16x16'},    # Low-res branch with adaptive pooling
        {'source_layer': 0, 'refinements': 1, 'pooling': 8},          # Extra low-res branch
    ]

    # Create a dummy input tensor
    dummy_input = torch.randn(4, 3, 64, 64) # (batch, channels, height, width)
    
    # Instantiate the model with the custom configuration
    model = ConfigurableHighwayNet(
        in_channels=3,
        num_classes=10,
        branch_configs=custom_config,
        initial_channels=32,
        mlp_hidden_dim=256
    )
    
    # Pass the dummy input through the model
    print("Testing model with a custom configuration...")
    output = model(dummy_input)
    
    # Print the output shape to verify
    print("\nModel instantiated and tested successfully!")
    print(f"Input shape:  {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected output shape: (4, 10)")
