import torch
from torchinfo import summary
from src.model import Network
from src.utils import load_config

# Load config
config = load_config("config/config.yaml")

# Setup device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize model with same parameters as in main.py
INPUT_SIZE = (2 * config["context"] + 1) * 28
model = Network(INPUT_SIZE, 42, config).to(device)

# Print model summary
summary(model, input_size=(1, 2*config["context"] + 1, 28))