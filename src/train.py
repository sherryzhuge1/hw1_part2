import gc
import os

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

import wandb


def train(model, dataloader, optimizer, scheduler, criterion, config, device, scaler, epoch):
    model.train()
    tloss, tacc = 0, 0
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="Train")

    for i, (frames, phonemes) in enumerate(dataloader):
        optimizer.zero_grad()

        frames = frames.to(device, non_blocking=True)
        phonemes = phonemes.to(device, non_blocking=True)

        with torch.autocast(device_type=device, dtype=torch.float16):
            logits1 = model(frames)
            logits2 = model(frames)

            loss1 = criterion(logits1, phonemes)
            loss2 = criterion(logits2, phonemes)
            loss_nll = loss1 + loss2

            log_p1 = F.log_softmax(logits1, dim=1)
            p2 = F.softmax(logits2, dim=1)
            kl_loss = 0.5 * (
                F.kl_div(log_p1, p2, reduction="batchmean")
                + F.kl_div(
                    F.log_softmax(logits2, dim=1),
                    F.softmax(logits1, dim=1),
                    reduction="batchmean",
                )
            )

            total_loss = loss_nll + config["r_drop_alpha"] * kl_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            avg_logits = (logits1 + logits2) / 2
            acc = torch.sum(torch.argmax(avg_logits, dim=1) == phonemes).item() / phonemes.size(0)

        tloss += total_loss.item()
        tacc += acc

        wandb.log(
            {
                "train_batch_loss": total_loss.item(),
                "train_batch_acc": acc,
                "train_batch": i + 1,
            }
        )

        batch_bar.set_postfix(
            loss=f"{float(tloss / (i + 1)):.04f}",
            acc=f"{float(tacc * 100 / (i + 1)):.04f}%",
        )
        batch_bar.update()

        del frames, phonemes, logits1, logits2
        if i % 50 == 0:
            torch.cuda.empty_cache()

    batch_bar.close()
    return tloss / len(dataloader), tacc / len(dataloader)


def eval(model, dataloader, criterion, device):
    model.eval()
    vloss, vacc = 0, 0
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc="Val")

    for i, (frames, phonemes) in enumerate(dataloader):
        frames = frames.to(device)
        phonemes = phonemes.to(device)

        with torch.inference_mode():
            logits = model(frames)
            loss = criterion(logits, phonemes)

        vloss += loss.item()
        vacc += torch.sum(torch.argmax(logits, dim=1) == phonemes).item() / logits.shape[0]

        wandb.log(
            {
                "val_batch_loss": loss.item(),
                "val_batch_acc": torch.sum(torch.argmax(logits, dim=1) == phonemes).item() / logits.shape[0],
                "val_batch": i + 1,
            }
        )

        batch_bar.set_postfix(
            loss="{:.04f}".format(float(vloss / (i + 1))),
            acc="{:.04f}%".format(float(vacc * 100 / (i + 1))),
        )
        batch_bar.update()

        del frames, phonemes, logits
        torch.cuda.empty_cache()

    batch_bar.close()
    return vloss / len(dataloader), vacc / len(dataloader)


def test(model, test_loader, device, phonemes):
    model.eval()
    predictions = []

    with torch.no_grad():
        for frames in test_loader:
            frames = frames.to(device)
            logits = model(frames)
            preds = torch.argmax(logits, dim=1)
            predictions.extend([phonemes[p] for p in preds.cpu().numpy()])

    return predictions


def train_model(
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
):
    torch.cuda.empty_cache()
    gc.collect()

    # Initialize tracking variables
    start_epoch = 0
    best_val_acc = 0
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load best model from wandb if it exists
    try:
        # Get the best model artifact from wandb
        api = wandb.Api()
        artifact = api.artifact(f"{wandb.run.entity}/{wandb.run.project}/model-{wandb.run.id}:latest")

        # Download the model file
        artifact_dir = artifact.download()
        checkpoint_path = os.path.join(artifact_dir, "best_model.pth")

        print("Loading checkpoint from wandb...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_acc = checkpoint["val_acc"]
        print(f"Resuming from epoch {start_epoch} with best validation accuracy: {best_val_acc * 100:.2f}%")
    except Exception as e:
        print(f"No checkpoint found on wandb or error loading checkpoint: {e}")
        print("Starting training from scratch...")

    wandb.watch(model, log="all")

    for epoch in range(start_epoch, config["epochs"]):
        print(f"\nEpoch {epoch + 1}/{config['epochs']}")

        curr_lr = float(optimizer.param_groups[0]["lr"])
        train_loss, train_acc = train(
            model, train_loader, optimizer, scheduler, criterion, config, device, scaler, epoch
        )

        val_loss, val_acc = eval(model, val_loader, criterion, device)

        print(f"\tTrain Acc {train_acc * 100:.04f}%\tTrain Loss {train_loss:.04f}\tLearning Rate {curr_lr:.07f}")
        print(f"\tVal Acc {val_acc * 100:.04f}%\tVal Loss {val_loss:.04f}")

        wandb.log(
            {
                "train_acc": train_acc * 100,
                "train_loss": train_loss,
                "val_acc": val_acc * 100,
                "valid_loss": val_loss,
                "lr": curr_lr,
            }
        )

        scheduler.step()

        if val_acc > best_val_acc:
            # Delete previous best checkpoint if it exists
            if best_val_acc > 0:  # Only if we had a previous best
                try:
                    # Delete local checkpoint
                    prev_checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
                    if os.path.exists(prev_checkpoint_path):
                        os.remove(prev_checkpoint_path)

                    # Delete wandb checkpoint
                    api = wandb.Api()
                    prev_artifact = api.artifact(f"{wandb.run.entity}/{wandb.run.project}/model-{wandb.run.id}:best")
                    prev_artifact.delete()
                except Exception as e:
                    print(f"Error deleting previous checkpoint: {e}")

            # Save new best checkpoint
            best_val_acc = val_acc
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "config": config,
            }

            # Save locally first
            checkpoint_path = os.path.join(checkpoint_dir, "best_model.pth")
            torch.save(checkpoint, checkpoint_path)

            # Save to wandb
            wandb.save(checkpoint_path)

            # Log best model artifact
            artifact = wandb.Artifact(
                name=f"model-{wandb.run.id}",
                type="model",
                description=f"Best model at epoch {epoch} with val_acc {val_acc:.4f}",
                metadata={"val_acc": val_acc},
            )
            artifact.add_file(checkpoint_path)
            wandb.log_artifact(artifact, aliases=["best"])

    return best_val_acc
