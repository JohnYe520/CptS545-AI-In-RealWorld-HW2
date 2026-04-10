from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits, dim=1)


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def nll(logits: torch.Tensor, targets: torch.Tensor) -> float:
    return F.cross_entropy(logits, targets).item()


def ece(logits: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    probs = probs_from_logits(logits)
    confidences, predictions = probs.max(dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece_value = torch.zeros(1)

    # Compute expected calibration error
    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        prop = in_bin.float().mean()
        if prop.item() > 0:
            acc_bin = accuracies[in_bin].float().mean()
            conf_bin = confidences[in_bin].mean()
            ece_value += torch.abs(conf_bin - acc_bin) * prop

    return ece_value.item()


def risk_coverage_curve(logits: torch.Tensor, targets: torch.Tensor):
    probs = probs_from_logits(logits)
    confidences, preds = probs.max(dim=1)
    correct = preds.eq(targets).float()

    # Sort by confidence
    order = torch.argsort(confidences, descending=True)
    correct = correct[order]

    n = len(targets)
    coverages = []
    risks = []

    cumulative_correct = 0.0
    for k in range(1, n + 1):
        cumulative_correct += correct[k - 1].item()
        coverage = k / n
        risk = 1.0 - (cumulative_correct / k)
        coverages.append(coverage)
        risks.append(risk)

    return np.array(coverages), np.array(risks)