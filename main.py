from __future__ import annotations

import json
import os
import torch

from data import get_loaders
from model import build_model
from train import fit_model, load_model
from uq_methods import (
    baseline_softmax_confidence,
    deep_ensemble_method,
    mc_dropout_method,
    conformal_prediction,
)

epochs = 10

def train_or_load(
    ckpt_path,
    train_loader,
    test_loader,
    device,
    epochs=epochs,
    dropout_p=0.0,
):
    model = build_model(num_classes=100, dropout_p=dropout_p)

    if os.path.exists(ckpt_path):
        print(f"Loading {ckpt_path}")
        model = load_model(model, ckpt_path, device)
    else:
        print(f"Training {ckpt_path}")
        fit_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            device=device,
            epochs=epochs,
            save_path=ckpt_path,
        )
        model = load_model(model, ckpt_path, device)

    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Load data
    train_loader, calib_loader, test_loader = get_loaders(
        root="./data",
        batch_size=128,
        num_workers=2,
        calib_size=5000,
        seed=42,
    )

    # Baseline model
    baseline_model = train_or_load(
        "checkpoints/baseline.pth",
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        dropout_p=0.0,
    )

    baseline_results = baseline_softmax_confidence(
        baseline_model, test_loader, device
    )

    # Deep ensemble
    ensemble_models = []
    for i in range(3):
        model = train_or_load(
            f"checkpoints/ensemble_{i}.pth",
            train_loader,
            test_loader,
            device,
            epochs=epochs,
            dropout_p=0.0,
        )
        ensemble_models.append(model)

    ensemble_results = deep_ensemble_method(
        ensemble_models, test_loader, device
    )

    # MC dropout
    mc_model = train_or_load(
        "checkpoints/mc_dropout.pth",
        train_loader,
        test_loader,
        device,
        epochs=epochs,
        dropout_p=0.2,
    )

    mc_results = mc_dropout_method(
        mc_model, test_loader, device, mc_passes=5
    )

    # Conformal prediction
    conformal_results = conformal_prediction(
        baseline_model,
        calib_loader,
        test_loader,
        device,
        alpha=0.1,
    )

    # Save results
    results = {
        "baseline_softmax_confidence": baseline_results,
        "deep_ensemble": ensemble_results,
        "mc_dropout": mc_results,
        "conformal_prediction": conformal_results,
    }

    print(json.dumps(results, indent=2))

    with open("results/results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()