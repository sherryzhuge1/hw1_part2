import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self, input_size, output_size, config):
        super(Network, self).__init__()
        self.config = config

        self.model = nn.Sequential(
            # Input layer - no dropout
            nn.Linear(input_size, 3222),
            nn.LayerNorm(3222),
            nn.GELU(),
            # First hidden - dropout
            nn.Linear(3222, 2278),
            nn.LayerNorm(2278),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            # Second hidden - no dropout
            nn.Linear(2278, 1610),
            nn.LayerNorm(1610),
            nn.GELU(),
            # Third hidden - dropout
            nn.Linear(1610, 1139),
            nn.LayerNorm(1139),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            # Fourth hidden - no dropout
            nn.Linear(1139, 805),
            nn.LayerNorm(805),
            nn.GELU(),
            # Fifth hidden - dropout
            nn.Linear(805, 569),
            nn.LayerNorm(569),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            # Sixth hidden - no dropout
            nn.Linear(569, 402),
            nn.LayerNorm(402),
            nn.GELU(),
            # Output layer
            nn.Linear(402, output_size),
        )

        if config["weight_initialization"] is not None:
            self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="leaky_relu", a=0.01)
                m.bias.data.fill_(0)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        return self.model(x)
