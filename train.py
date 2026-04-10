from __future__ import annotations

import os
import torch
import torch.nn as nn


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def get_logits_and_targets(model, loader, device):
    # Get logits on a loader
    model.eval()
    all_logits = []
    all_targets = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images)
        all_logits.append(logits.cpu())
        all_targets.append(targets)

    return torch.cat(all_logits, dim=0), torch.cat(all_targets, dim=0)


def fit_model(
    model,
    train_loader,
    test_loader,
    device,
    epochs=30,
    lr=0.1,
    weight_decay=5e-4,
    save_path=None,
):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.to(device)
    best_acc = 0.0

    for epoch in range(epochs):
        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        # Evaluate on test set
        test_logits, test_targets = get_logits_and_targets(model, test_loader, device)
        test_acc = (test_logits.argmax(dim=1) == test_targets).float().mean().item()

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"train_loss={train_loss:.4f} "
            f"train_acc={train_acc:.4f} "
            f"test_acc={test_acc:.4f}"
        )

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            if save_path is not None:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                torch.save(model.state_dict(), save_path)


def load_model(model, ckpt_path, device):
    # Load saved checkpoint
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model