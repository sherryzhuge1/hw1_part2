import os
import subprocess

import torch
import yaml

import wandb
from src.dataset import create_dataloaders
from src.model import Network
from src.train import train_model
from src.utils import PHONEMES, load_config


def setup_kaggle_data():
    """Setup Kaggle data if not already present"""
    if not os.path.exists("11785-s25-hw1p2"):
        # Setup Kaggle credentials
        kaggle_path = os.path.expanduser("~/.kaggle")
        kaggle_json = os.path.join(kaggle_path, "kaggle.json")

        # Create .kaggle directory if it doesn't exist
        os.makedirs(kaggle_path, exist_ok=True)

        # Create kaggle.json if it doesn't exist
        if not os.path.exists(kaggle_json):
            with open(kaggle_json, "w") as f:
                f.write(
                    '{"username":"clement6","key":"11f8da07ebaf64f72403614544126328"}'
                )
            # Set proper permissions
            os.chmod(kaggle_json, 0o600)

        # Download and extract data
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", "11785-spring-25-hw-1-p-2"],
            check=True,
        )
        subprocess.run(["unzip", "-qn", "11785-spring-25-hw-1-p-2.zip"], check=True)


def update_model_architecture(context):
    """Run optimize_model.py with new context and update model.py"""
    # Update config
    config = load_config("config/config.yaml")
    config["context"] = context

    # Save updated config
    with open("config/config.yaml", "w") as f:
        yaml.dump(config, f)

    # Run optimize_model.py to get new architecture
    subprocess.run(["python", "optimize_model.py"], check=True)


def main():
    # Setup data first
    setup_kaggle_data()

    # Initialize wandb
    wandb.init()

    # Load base config
    config = load_config("config/config.yaml")

    # Update config with sweep values
    for key, value in wandb.config.items():
        config[key] = value

    # Update model architecture if context changed
    update_model_architecture(config["context"])

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Setup data
    ROOT = "11785-s25-hw1p2"
    train_loader, val_loader, test_loader = create_dataloaders(config, ROOT)

    # Initialize model
    INPUT_SIZE = (2 * config["context"] + 1) * 28
    model = Network(INPUT_SIZE, len(PHONEMES), config).to(device)
    model = torch.compile(model)

    # Setup training
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["learning_rate"],
        steps_per_epoch=len(train_loader),
        epochs=config["epochs"],
        pct_start=0.15,
        div_factor=25,
        final_div_factor=1000,
    )

    scaler = torch.amp.GradScaler(enabled=True)

    # Training loop
    _ = train_model(
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        criterion,
        config,
        device,
        wandb.run,
        scaler,
    )


if __name__ == "__main__":
    main()
