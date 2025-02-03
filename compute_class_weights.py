import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.dataset import create_dataloaders
from src.utils import PHONEMES, load_config


def compute_class_weights():
    # Load config
    config = load_config("config/config.yaml")

    # Create dataloaders
    ROOT = "11785-s25-hw1p2"
    train_loader, _, _ = create_dataloaders(config, ROOT)

    # Initialize counters - exclude [SOS] and [EOS]
    num_real_phonemes = len(PHONEMES) - 2  # Subtract 2 for [SOS] and [EOS]
    class_counts = torch.zeros(num_real_phonemes)
    total_samples = 0

    print("Counting class frequencies...")
    for _, labels in tqdm(train_loader, desc="Computing class weights"):
        for label in labels:
            if label < num_real_phonemes:  # Only count actual phonemes
                class_counts[label] += 1
                total_samples += 1

    # Calculate class frequencies
    class_frequencies = class_counts / total_samples

    # Calculate smoothed weights using square root
    beta = 0.5  # Smoothing factor (0 = no weighting, 1 = full weighting)
    weights = 1.0 / torch.sqrt(class_frequencies)  # Square root makes the weights less extreme
    weights = beta * weights + (1 - beta)  # Smooth the weights towards 1.0
    weights = weights / weights.mean()  # Normalize so mean weight is 1

    print("\nClass statistics:")
    for idx, (phoneme, freq, weight) in enumerate(zip(PHONEMES[:-2], class_frequencies, weights)):
        print(f"{phoneme}: freq={freq:.4f} ({int(class_counts[idx])} samples), weight={weight:.4f}")

    # Plot distributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

    # Plot frequencies
    ax1.bar(PHONEMES[:-2], class_frequencies.numpy())
    ax1.set_title("Phoneme Distribution in Training Set")
    ax1.set_xticklabels(PHONEMES[:-2], rotation=45)
    ax1.set_ylabel("Frequency")

    # Plot weights
    ax2.bar(PHONEMES[:-2], weights.numpy())
    ax2.set_title("Class Weights")
    ax2.set_xticklabels(PHONEMES[:-2], rotation=45)
    ax2.set_ylabel("Weight")

    plt.tight_layout()
    plt.savefig("phoneme_distribution.png")

    # Add zero weights for [SOS] and [EOS]
    weights = torch.cat([weights, torch.zeros(2)])

    # Save weights
    torch.save(weights, "focal_loss_weights.pt")
    print("\nWeights saved to 'focal_loss_weights.pt'")

    return weights


if __name__ == "__main__":
    weights = compute_class_weights()
