from __future__ import annotations

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# CIFAR-100 normalization constants
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)

# Data loading and preprocessing
def get_loaders(
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    calib_size: int = 5000,
    seed: int = 42,
):
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
    ])

    # Load CIFAR-100
    train_full_aug = datasets.CIFAR100(
        root=root, train=True, download=True, transform=train_transform
    )
    train_full_eval = datasets.CIFAR100(
        root=root, train=True, download=False, transform=test_transform
    )
    test_set = datasets.CIFAR100(
        root=root, train=False, download=True, transform=test_transform
    )

    # Split train and calibration sets
    train_size = len(train_full_aug) - calib_size
    generator = torch.Generator().manual_seed(seed)

    train_set, _ = random_split(
        train_full_aug, [train_size, calib_size], generator=generator
    )
    _, calib_set = random_split(
        train_full_eval,
        [train_size, calib_size],
        generator=torch.Generator().manual_seed(seed),
    )

    # Build loaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    calib_loader = DataLoader(
        calib_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, calib_loader, test_loader