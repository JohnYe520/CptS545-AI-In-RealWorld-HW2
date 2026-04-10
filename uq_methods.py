from __future__ import annotations

import torch
import torch.nn.functional as F

from metrics import accuracy, ece, nll, risk_coverage_curve
from train import get_logits_and_targets

from torch_uncertainty.models.wrappers import deep_ensembles, mc_dropout
from torchcp.classification.predictor import SplitPredictor
from torchcp.classification.score import APS


def baseline_softmax_confidence(model, test_loader, device):
    # Baseline softmax confidence
    logits, targets = get_logits_and_targets(model, test_loader, device)
    coverage, risk = risk_coverage_curve(logits, targets)

    return {
        "accuracy": accuracy(logits, targets),
        "ece": ece(logits, targets),
        "nll": nll(logits, targets),
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
    }


@torch.no_grad()
def deep_ensemble_method(models, test_loader, device):
    # Deep ensemble from TorchUncertainty
    ensemble_model = deep_ensembles(models)

    # Move models to device
    ensemble_model.to(device)
    for model in models:
        model.to(device)
        model.eval()

    all_probs = []
    targets_ref = None

    # Average ensemble probabilities
    for model in models:
        logits, targets = get_logits_and_targets(model, test_loader, device)
        probs = F.softmax(logits, dim=1)
        all_probs.append(probs)
        if targets_ref is None:
            targets_ref = targets

    mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    ensemble_logits = torch.log(mean_probs.clamp_min(1e-12))
    coverage, risk = risk_coverage_curve(ensemble_logits, targets_ref)

    return {
        "accuracy": accuracy(ensemble_logits, targets_ref),
        "ece": ece(ensemble_logits, targets_ref),
        "nll": nll(ensemble_logits, targets_ref),
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
    }


@torch.no_grad()
def mc_dropout_method(model, test_loader, device, mc_passes=20):
    # MC dropout from TorchUncertainty
    model.to(device)
    wrapped_model = mc_dropout(model, num_estimators=mc_passes)
    wrapped_model.to(device)
    wrapped_model.eval()

    all_probs = []
    all_targets = []

    for images, targets in test_loader:
        images = images.to(device, non_blocking=True)

        # Get stochastic predictions
        logits = wrapped_model(images)

        # Reshape if outputs are stacked
        if logits.dim() == 2:
            batch_size = images.size(0)
            logits = logits.view(mc_passes, batch_size, -1)

        probs = F.softmax(logits, dim=-1).mean(dim=0)

        all_probs.append(probs.cpu())
        all_targets.append(targets)

    mean_probs = torch.cat(all_probs, dim=0)
    targets = torch.cat(all_targets, dim=0)
    mc_logits = torch.log(mean_probs.clamp_min(1e-12))
    coverage, risk = risk_coverage_curve(mc_logits, targets)

    return {
        "accuracy": accuracy(mc_logits, targets),
        "ece": ece(mc_logits, targets),
        "nll": nll(mc_logits, targets),
        "coverage": coverage.tolist(),
        "risk": risk.tolist(),
    }


def conformal_prediction(model, calib_loader, test_loader, device, alpha=0.1):
    # Split conformal prediction
    model.to(device)
    model.eval()

    predictor = SplitPredictor(
        score_function=APS(),
        model=model,
        alpha=alpha,
        device=device,
    )

    # Calibrate and evaluate
    predictor.calibrate(calib_loader)
    results = predictor.evaluate(test_loader)

    return {
        "set_coverage": float(results["coverage_rate"]),
        "avg_set_size": float(results["average_size"]),
    }