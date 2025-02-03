import math

from src.utils import load_config


def count_parameters(base_width, input_size):
    """Calculate total parameters for a model with given base width and sqrt(2) reduction ratio"""
    total_params = 0
    ratio = math.sqrt(2)  # reduction ratio between layers

    # Calculate layer widths using the ratio
    widths = [
        int(base_width),
        int(base_width / ratio),
        int(base_width / ratio**2),
        int(base_width / ratio**3),
        int(base_width / ratio**4),
        int(base_width / ratio**5),
        int(base_width / ratio**6),
    ]

    # First layer: input_size -> widths[0]
    total_params += (input_size + 1) * widths[0]  # weights + biases
    total_params += 2 * widths[0]  # LayerNorm params

    # Hidden layers
    for i in range(len(widths) - 1):
        # Linear layer
        total_params += (widths[i] + 1) * widths[i + 1]  # weights + biases
        # LayerNorm params
        total_params += 2 * widths[i + 1]

    # Output layer (42 classes)
    total_params += (widths[-1] + 1) * 42

    return total_params, widths


def main():
    # Load config to get context
    config = load_config("config/config.yaml")
    context = config["context"]

    # Calculate input size based on context
    input_size = (2 * context + 1) * 28

    target_params = 20_000_000

    # Binary search to find optimal base width
    left, right = 100, 5000
    best_base = None
    best_params = float("inf")
    best_widths = None

    while right - left > 1:
        mid = (left + right) // 2
        params, widths = count_parameters(mid, input_size)

        print(f"Base width: {mid:4d}, Params: {params:,}, Target: {target_params:,}")

        if params <= target_params and abs(target_params - params) < abs(
            target_params - best_params
        ):
            best_base = mid
            best_params = params
            best_widths = widths

        if params > target_params:
            right = mid
        else:
            left = mid

    if best_base is None:
        raise ValueError("No solution found under target parameters!")

    print(f"\nOptimal solution found for context {context}:")
    print(f"Base width: {best_base}")
    print(f"Total parameters: {best_params:,}")
    print(f"Layer widths: {best_widths}")

    # Generate model.py content
    model_template = f"""import torch
import torch.nn as nn

class Network(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(Network, self).__init__()
        self.config = config
        
        self.model = nn.Sequential(
            # Input layer - no dropout
            nn.Linear(input_size, {best_widths[0]}),
            nn.LayerNorm({best_widths[0]}),
            nn.GELU(),
            
            # First hidden - dropout
            nn.Linear({best_widths[0]}, {best_widths[1]}),
            nn.LayerNorm({best_widths[1]}),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            
            # Second hidden - no dropout
            nn.Linear({best_widths[1]}, {best_widths[2]}),
            nn.LayerNorm({best_widths[2]}),
            nn.GELU(),
            
            # Third hidden - dropout
            nn.Linear({best_widths[2]}, {best_widths[3]}),
            nn.LayerNorm({best_widths[3]}),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            
            # Fourth hidden - no dropout
            nn.Linear({best_widths[3]}, {best_widths[4]}),
            nn.LayerNorm({best_widths[4]}),
            nn.GELU(),
            
            # Fifth hidden - dropout
            nn.Linear({best_widths[4]}, {best_widths[5]}),
            nn.LayerNorm({best_widths[5]}),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            
            # Sixth hidden - no dropout
            nn.Linear({best_widths[5]}, {best_widths[6]}),
            nn.LayerNorm({best_widths[6]}),
            nn.GELU(),
            
            # Output layer
            nn.Linear({best_widths[6]}, output_size)
        )
        
        if config["weight_initialization"] is not None:
            self.initialize_weights()
    
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01
                )
                m.bias.data.fill_(0)
    
    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.model(x)
"""

    # Write the new model.py
    with open("src/model.py", "w") as f:
        f.write(model_template)


if __name__ == "__main__":
    main()
