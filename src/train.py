#!/usr/bin/env python3
"""
Training script for Loan Approval Prediction model.
Run: python src/train.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model import LoanApprovalModel

def main():
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION — MODEL TRAINING")
    print("=" * 60)

    pipeline = LoanApprovalModel(models_dir='models')

    # Load / generate data
    pipeline.load_data(n_samples=10240)

    # Preprocess
    pipeline.preprocess(test_size=0.2)

    # Train all models
    pipeline.train_all()

    # Evaluate
    pipeline.evaluate()

    # Save
    pipeline.save_models()

    print("\n" + "=" * 60)
    print("Training complete! Models saved to ./models/")
    print("=" * 60)

if __name__ == "__main__":
    main()
