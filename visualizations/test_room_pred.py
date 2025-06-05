import os
import yaml
from attrdict import AttrDict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import csv

# Import the custom model and dataset modules.
from modules.semantic.semantic_net_pl_maskformer_small_room_type import semantic_net_pl_maskformer_small_room_type
from data_utils.data_utils import GridSeqDataset

def main():
    # ----------------------
    # Hardcoded configuration
    # ----------------------
    config = {
        "dataset": "test_data_set_full",
        "dataset_path": "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/structured3d_perspective/test_data_set_full",
        "ckpt_path": "./logs/room_type_no_backbone/",
        "epochs": 100,
        "batch_size": 1,
        "lr": 0.001,
        "shape_loss_weight": None,
        "num_classes": 4,  # For semantic segmentation (if used)
        "depth_net": {
            "L": 0,
            "D": 128,
            "d_min": 0.1,
            "d_max": 15.0,
            "d_hyp": -0.2,
            "F_W": 3 / 5,  # Evaluates to 0.6
        },
        "augmentation": {
            "add_rp": True,
            "roll": 0,
            "pitch": 0,
        },
        "use_maskformer": True,
        "use_small": True,
        "model_type": "semantic",
        "image_augment": False,
        "image_augment_noise_std": 0.02,
        "use_room_type": True,
        "num_room_types": 16,  # Use this for room type prediction.
        "room_type_only": False,
        "room_type_no_backbone": True,
        "log_dir_semantic_and_room_aware":  '/home/yuvalg/projects/Semantic_Floor_plan_localization/modules/final_weights/semantic/semantic_net-epoch=14-loss-valid=1.31.ckpt',
        "semantic_net_type": "resnet_cls",
        "room_data_dir": '/home/yuvalg/projects/Semantic_Floor_plan_localization/data/structured3d_perspective/test_data_set_full'
    }

    # ----------------------
    # Hardcoded checkpoint path
    # ----------------------
    ckpt_file = config['log_dir_semantic_and_room_aware']

    # ----------------------
    # Set up the test dataset and DataLoader
    # ----------------------
    dataset_path = config["dataset_path"]
    split_file = os.path.join(dataset_path, "split.yaml")
    if not os.path.exists(split_file):
        raise FileNotFoundError(f"Could not find split file at {split_file}")
    
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    test_split = split.get("test", split.val)[:-1]

    test_set = GridSeqDataset(
        dataset_dir=dataset_path,
        scene_names=test_split,
        L=0,
        room_data_dir=config["room_data_dir"],     
    )

    test_loader = DataLoader(
        test_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=8
    )

    # ----------------------
    # Load the model from the checkpoint
    # ----------------------
    model = semantic_net_pl_maskformer_small_room_type.load_from_checkpoint(
        checkpoint_path=ckpt_file,
        num_classes=config["num_classes"],
        semantic_net_type=config["semantic_net_type"],
        num_room_types=config["num_room_types"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    # ----------------------
    # Prepare metrics and storage
    # ----------------------
    total_loss = 0.0
    total_acc = 0.0
    batch_count = 0
    total_samples = 0
    top2_correct = 0
    top5_correct = 0

    correct_confidences = []
    incorrect_confidences = []
    all_confidences = []
    all_correct_flags = []
    results = []  # store per-sample results

    # ----------------------
    # Evaluation loop with progress bar
    # ----------------------
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(test_loader, desc="Evaluating", unit="sample")):
            images = batch["ref_img"].to(device)
            room_labels = batch["room_label"].long().to(device)
            _, room_logits, _ = model(images)
            loss = F.cross_entropy(room_logits, room_labels)
            total_loss += loss.item()

            probs = F.softmax(room_logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            batch_acc = (preds == room_labels).float().mean().item()
            total_acc += batch_acc

            batch_size = room_labels.size(0)
            total_samples += batch_size
            batch_count += 1

            # Top-2 and top-5
            _, top2 = torch.topk(probs, k=2, dim=1)
            _, top5 = torch.topk(probs, k=5, dim=1)
            top2_hits = (top2 == room_labels.unsqueeze(1)).any(dim=1).float()
            top5_hits = (top5 == room_labels.unsqueeze(1)).any(dim=1).float()
            top2_correct += top2_hits.sum().item()
            top5_correct += top5_hits.sum().item()

            # record results per sample
            for i in range(batch_size):
                pred = preds[i].item()
                gt = room_labels[i].item()
                conf = max_probs[i].item()
                correct = int(pred == gt)
                all_confidences.append(conf)
                all_correct_flags.append(correct)
                if correct:
                    correct_confidences.append(conf)
                else:
                    incorrect_confidences.append(conf)
                results.append({
                    'index': idx * config['batch_size'] + i,
                    'predicted': pred,
                    'ground_truth': gt,
                    'confidence': conf,
                    'correct': correct
                })

    # Compute overall statistics
    avg_loss = total_loss / batch_count
    avg_acc = total_acc / batch_count
    top2_acc = top2_correct / total_samples
    top5_acc = top5_correct / total_samples

    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test Accuracy: {avg_acc:.4f}")
    print(f"Top-2 Accuracy: {top2_acc:.4f}")
    print(f"Top-5 Accuracy: {top5_acc:.4f}\n")

    # Compute wrong high-confidence samples
    thresh = 0.8
    wrong_high_conf = [r for r in results if r['correct'] == 0 and r['confidence'] >= thresh]
    num_wrong_high = len(wrong_high_conf)
    pct_wrong_high = num_wrong_high / total_samples * 100
    print(f"Number of wrong samples with confidence >= {thresh*100:.0f}%: {num_wrong_high}")
    print(f"Percentage of such samples: {pct_wrong_high:.2f}%\n")

    # Optimal threshold analysis
    thresholds = np.linspace(0, 1, 101)
    all_conf_np = np.array(all_confidences)
    all_corr_np = np.array(all_correct_flags)
    best_thresh = 0.0
    best_metric = -1.0
    best_acc_t = 0.0
    best_cov = 0.0
    for t in thresholds:
        mask = all_conf_np >= t
        if not mask.any():
            continue
        cov = mask.sum() / len(all_conf_np)
        acc_t = all_corr_np[mask].mean()
        metric = cov * acc_t
        if metric > best_metric:
            best_metric = metric
            best_thresh = t
            best_acc_t = acc_t
            best_cov = cov
    print("Optimal Prediction Threshold Analysis:")
    print(f"  Optimal threshold: {best_thresh:.2f}")
    print(f"  Coverage at optimal threshold: {best_cov*100:.2f}%")
    print(f"  Accuracy at optimal threshold: {best_acc_t*100:.2f}%\n")

    # Save results CSV
    save_dir = "/home/yuvalg/projects/Semantic_Floor_plan_localization/results/visualizations/room_analysis"
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, "room_pred_results.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['index', 'predicted', 'ground_truth', 'confidence', 'correct'])
        writer.writeheader()
        for r in results:
            writer.writerow(r)
    print(f"Per-sample results saved to {csv_path}")

    # Plot the distributions
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(correct_confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Confidence for Correct Predictions')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(incorrect_confidences, bins=20, alpha=0.7, edgecolor='black')
    plt.title('Confidence for Misclassifications')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "room_pred_confidences.png")
    plt.savefig(plot_path)
    print(f"Plots saved to {plot_path}")

    # Optionally show plots
    # plt.show()

if __name__ == "__main__":
    main()
