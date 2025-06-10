import numpy as np
import json
import os

def calculate_recalls(trans_errors, rot_errors):
    """
    Calculates recall metrics based on translation and rotation errors.
    Handles empty arrays to avoid runtime warnings.
    """
    if trans_errors.size == 0 or rot_errors.size == 0:
        return {'1m': 0.0, '0.5m': 0.0, '0.1m': 0.0, '1m 30 deg': 0.0}

    return {
        "1m": np.sum(trans_errors < 1) / trans_errors.shape[0],
        "0.5m": np.sum(trans_errors < 0.5) / trans_errors.shape[0],
        "0.1m": np.sum(trans_errors < 0.1) / trans_errors.shape[0],
        "1m 30 deg": np.sum(np.logical_and(trans_errors < 1, rot_errors < 30)) / trans_errors.shape[0],
    }

def save_and_print_results(
    results_dir,
    weight_key,
    baseline_recalls,
    baseline_recalls_n,
    refine_recalls,
    refine_recalls_n
):
    """Saves recall results to JSON files and prints them."""
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, f"baseline_recalls_{weight_key}.json"), "w") as f:
        json.dump(baseline_recalls, f, indent=4)
    with open(os.path.join(results_dir, f"baseline_recalls_n_{weight_key}.json"), "w") as f:
        json.dump(baseline_recalls_n, f, indent=4)
    with open(os.path.join(results_dir, f"refine_recalls_{weight_key}.json"), "w") as f:
        json.dump(refine_recalls, f, indent=4)
    with open(os.path.join(results_dir, f"refine_recalls_n_{weight_key}.json"), "w") as f:
        json.dump(refine_recalls_n, f, indent=4)

    print(f"Weight {weight_key} recalls:")
    print("  baseline:", baseline_recalls)
    print("  baseline_n:", baseline_recalls_n)
    print("  refine:", refine_recalls)
    print("  refine_n:", refine_recalls_n) 