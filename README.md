# CPTS580-HW2

## Overview
This is HW2 for **CPTS 545: AI in Real-World** course.

## Project Structure
```text
HW2/
├── main.py                      # Main script to run training and evaluation
├── data.py                      # Data loading (CIFAR-100)
├── model.py                     # ResNet-18 model
├── train.py                     # Training utilities
├── uq_methods.py                # UQ method implementations
├── metrics.py                   # Evaluation metrics
│
├── result_table.py              # Convert results.json to readable table
│
├── *.png                        # Result figures (training outputs)
│
# (Generated after running)
checkpoints/                     # Saved models (not included)
results/                         # Output JSON results (not included)
│
└── README.md                    # Instructions for running the code
```

## Requirements

Before running the project, make sure your Python version is 3.12 or higher.

Install the required Python libraries:

```bash
pip install torch torchvision numpy matplotlib
pip install lightning
pip install transformers==4.41.2 peft==0.10.0
pip install torch-uncertainty
pip install torchcp
```

## How to Run
```bash
# Step 1: Run the main script
python main.py
```

```text
This will:
    - Train all models
    - Evaluate all UQ methods
    - Generate: results/results.json
```

```bash
# Step 2: Generate result table
python result_table.py
```

```text
This script reads results.json and prints:
    - Accuracy
    - ECE
    - NLL
    - Conformal coverage
    - Conformal set size
```

## Result 
Final results are stored in:
```text
results/results.json
```

## Notes
```text
Training may take several hours depending on hardware.
```
