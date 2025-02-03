import os

import torch

from src.dataset import create_dataloaders
from src.losses import FocalLoss
from src.model import Network
from src.train import test, train_model
from src.utils import PHONEMES, load_config, save_predictions, setup_wandb


def main():
    # Load config
    config = load_config("config/config.yaml")

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
    criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=4
    )
    
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=31, eta_min=3e-4
    )
    
    first_plateau_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=3e-4 / 7.5e-4, total_iters=10
    )
    
    second_plateau_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=2e-4 / 7.5e-4, total_iters=10
    )
    
    third_plateau_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, factor=1e-4 / 7.5e-4, total_iters=10
    )
    
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler, first_plateau_scheduler, second_plateau_scheduler, third_plateau_scheduler],
        milestones=[4, 35, 45, 55]
    )

    scaler = torch.amp.GradScaler(enabled=True)

    # Initialize wandb
    run = setup_wandb(config)

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
        run,
        scaler,
    )

    # Load best model for testing
    checkpoint = torch.load(os.path.join("checkpoints", "best_model.pth"))
    model.load_state_dict(checkpoint["model_state_dict"])

    # Generate predictions
    predictions = test(model, test_loader, device, PHONEMES)
    save_predictions(predictions)

    run.finish()


if __name__ == "__main__":
    main()
