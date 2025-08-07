class PyTorchConvNode(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super(PyTorchConvNode, self).__init__()
        # 'same' padding in PyTorch for Conv2d
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
        
        # The shortcut projection handles spatial and channel alignment
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

        self.mlc_layers = nn.ModuleList()
        current_in_channels = input_channels

        for i in range(num_mlc_layers):
            conv_nodes_in_layer = nn.ModuleList()
            filters = initial_filters * (2 ** i)
            for j in range(num_conv_nodes_per_layer):
                conv_nodes_in_layer.append(
                    PyTorchConvNode(current_in_channels, filters, kernel_size[0])
                )
            self.mlc_layers.append(conv_nodes_in_layer)
            current_in_channels = filters * num_conv_nodes_per_layer

        self.flatten = nn.Flatten()
        # MLP layers are created dynamically in the forward pass
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

        if self.mlp is None:
            mlp_input_features = flatten_output.shape[1]
            mlp_layers = []
            in_features = mlp_input_features
            for units in self.mlp_units:
                mlp_layers.append(nn.Linear(in_features, units))
                mlp_layers.append(nn.ReLU())
                in_features = units
            self.mlp = nn.Sequential(*mlp_layers).to(x.device)
            self.output_layer = nn.Linear(in_features, self.output_classes).to(x.device)
        
        mlp_output = self.mlp(flatten_output)
        final_output = self.output_layer(mlp_output)
        return final_output
