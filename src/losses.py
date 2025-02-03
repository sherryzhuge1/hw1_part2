import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction="mean"):
        """
        Focal Loss implementation

        Args:
            gamma (float): Focusing parameter. Increases focus on hard examples.
                         Higher values mean more focus on hard examples.
            alpha (torch.Tensor, optional): Weight for each class. Addresses class imbalance.
                                          Should be of length num_classes.
            reduction (str): 'mean', 'sum' or 'none'. Default: 'mean'
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Model predictions (B, C), where B is batch size and C is number of classes
            targets: Ground truth labels (B,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)  # probability of the target class
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss
