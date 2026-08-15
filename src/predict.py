#!/usr/bin/env python3
"""
Inference script for single or batch predictions.
Run: python src/predict.py
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from model import LoanApprovalModel

SAMPLE_APPLICATION = {
    "annual_income": 75000,
    "loan_amount": 20000,
    "credit_score": 720,
    "employment_length": 5,
    "debt_to_income": 25.0,
    "home_ownership": "mortgage",
    "loan_purpose": "debt_consolidation",
    "previous_defaults": 0,
    "interest_rate": 10.5
}

def main():
    print("=" * 60)
    print("LOAN APPROVAL PREDICTION — INFERENCE")
    print("=" * 60)

    pipeline = LoanApprovalModel(models_dir='models')
    pipeline.load_models()

    print("\nSample Application:")
    print(json.dumps(SAMPLE_APPLICATION, indent=2))

    results = pipeline.predict(SAMPLE_APPLICATION)

    print("\n--- Prediction Results ---")
    for model_name, result in results.items():
        status = "APPROVED" if result['approved'] else "REJECTED"
        print(f"{model_name.replace('_', ' ').title():20s}: {status:10s} "
              f"(prob: {result['probability']:.2%}, confidence: {result['confidence']:.2%})")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    main()
