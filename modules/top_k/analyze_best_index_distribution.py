import os
import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from collections import Counter
from attrdict import AttrDict
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as mtick

# Import your existing TopKDataset.
from top_k_dataset import TopKDataset


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return AttrDict(yaml.safe_load(f))


def load_split(split_file):
    """Load the dataset split YAML file."""
    with open(split_file, "r") as f:
        split = AttrDict(yaml.safe_load(f))
    return split


def process_scene_names(scene_list):
    """
    Process scene names.
    If 'floor' is in the scene name, leave it as is; otherwise, assume a format like "scene_00000"
    and remove leading zeros.
    """
    processed = []
    for scene in scene_list:
        if 'floor' in scene:
            processed.append(scene)
        else:
            try:
                scene_number = int(scene.split('_')[1])
                scene_name = f"scene_{scene_number}"
            except Exception as e:
                print(f"Error processing scene {scene}: {e}")
                scene_name = scene
            processed.append(scene_name)
    return processed


def setup_output_paths(base_out_folder, args):
    """
    Set up the output file paths for all plots.

    Args:
        base_out_folder (str): Base directory to save plots.
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        dict: Dictionary with keys as plot names and values as their respective file paths.
    """
    os.makedirs(base_out_folder, exist_ok=True)
    return {
        "best_index_distribution": os.path.join(base_out_folder, args.plot_output),
        "distance_histogram": os.path.join(base_out_folder, args.distance_plot_output),
        "score_difference_histogram": os.path.join(base_out_folder, args.score_diff_plot_output),
        "semantic_depth_differences": os.path.join(base_out_folder, args.semantic_depth_plot_output),
    }


def plot_best_index_distribution(counter, output_path):
    """
    Plot a bar chart of best index distribution with percentages and raw counts annotated.

    Args:
        counter (Counter): Counter object with best index counts.
        output_path (str): File path to save the plot.
    """
    indices = sorted(counter.keys())
    counts = np.array([counter[idx] for idx in indices])
    total = counts.sum()
    percentages = counts / total * 100
    max_y = max(percentages) * 1.1

    df = pd.DataFrame({
        "Best Index": indices,
        "Count": counts,
        "Percentage": percentages
    })

    plt.figure(figsize=(10, 7))
    plot = sns.barplot(x="Best Index", y="Percentage", data=df, palette="Blues_d")
    for i, p in enumerate(plot.patches):
        plot.annotate(f"{df['Percentage'][i]:.1f}%\n({df['Count'][i]})",
                      (p.get_x() + p.get_width() / 2., p.get_height()),
                      ha="center", va="bottom", fontsize=10)

    plt.xlabel("K Index")
    plt.ylabel("Percentage (%)")
    plt.title("Distribution of Closest to Ground Truth K Index")
    plt.ylim(0, max_y)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Best Index distribution plot saved to {output_path}")
    plt.close()

def plot_distance_histogram(
    distances, 
    bins='auto', 
    bin_count=None, 
    output_path="distance_histogram.png"
):
    """
    Plot a histogram of distance differences with Y-axis as percentage of samples and bin range annotations.

    Args:
        distances (list): List of distance difference values.
        bins (int or array-like, optional): Number of bins or bin edges for the histogram. Defaults to 'auto'.
        bin_count (int, optional): If provided, overrides the 'bins' parameter to set the number of bins. Defaults to None.
        output_path (str): File path to save the plot. Defaults to "distance_histogram.png".
    """
    # Validate bin parameters
    if bin_count is not None:
        bins = bin_count

    # Create DataFrame
    df = pd.DataFrame({"Distance": distances})

    # Assign weights to normalize to percentage
    if len(distances) > 0:
        df['Weight'] = 100 / len(distances)
    else:
        df['Weight'] = 0  # Handle empty list

    # Initialize plot
    plt.figure(figsize=(10, 7))

    # Plot histogram with weights to represent percentages
    hist_plot = sns.histplot(
        data=df,
        x="Distance", 
        bins=bins, 
        weights="Weight", 
        kde=False, 
        color="coral", 
        edgecolor="black"
    )

    plt.xlabel("Difference in Distance (meters)")
    plt.ylabel("Percentage of Samples (%)")
    plt.title("Difference in Distance between nearest K's to GT vs. K0 to GT")

    # Format Y-axis to show percentage
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(xmax=100))

    # Adjust y-axis limit to accommodate up to 100%
    plt.ylim(0, 15)

    # Calculate bin counts and bin edges for annotations
    bin_counts, bin_edges = np.histogram(distances, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Annotate bin ranges above each bar
    for count, edge_left, edge_right in zip(bin_counts, bin_edges[:-1], bin_edges[1:]):
        bin_range = f"{edge_left:.2f}-{edge_right:.2f}"
        percentage = (count / len(distances)) * 100 if len(distances) > 0 else 0
        # Position the text slightly above the bar
        plt.text(
            (edge_left + edge_right) / 2, 
            percentage + 1.2,  # Offset by 1% above the bar
            bin_range,
            ha="center",
            va="center",
            fontsize=9,
            color="black",
            rotation=90,          # Rotate text by 90 degrees
            rotation_mode='anchor'  # Ensures rotation around the anchor point
        )

    # Adjust layout to ensure everything fits without overlapping
    plt.tight_layout()

    # Save the plot with tight bounding box
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Distance histogram plot saved to {output_path}")
    plt.close()

def plot_score_difference_histogram(score_diffs, bins, output_path):
    """
    Plot a histogram of score differences between top pick and K1 with annotations.

    Args:
        score_diffs (list): List of score difference values.
        bins (array-like): Bin edges for the histogram.
        output_path (str): File path to save the plot.
    """
    df = pd.DataFrame({"Score Difference": score_diffs})

    plt.figure(figsize=(10, 7))
    plot = sns.histplot(df['Score Difference'], bins=bins, kde=False, color="green", edgecolor="black")
    plt.xlabel("Localization Score Difference")
    plt.ylabel("Frequency")
    plt.title("Localization core Difference to K0 for Samples where Best Pick is not K0")

    # Annotate each bin with percentage and count.
    bin_counts, bin_edges = np.histogram(score_diffs, bins=bins)
    total = bin_counts.sum()
    percentages = bin_counts / total * 100
    max_y = max(bin_counts) * 1.2

    for i in range(len(bin_counts)):
        bin_range = f"{bin_edges[i]:.4f}-{bin_edges[i+1]:.4f}"
        plt.text((bin_edges[i] + bin_edges[i+1]) / 2, bin_counts[i] + 100,
                 f"{bin_range}\n{percentages[i]:.1f}%\n({bin_counts[i]})",
                 ha="center", va="center", fontsize=8,
                             rotation=90,
            rotation_mode='anchor')

    plt.ylim(0, max_y)
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Score difference histogram plot saved to {output_path}")
    plt.close()


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.ticker as mtick

def plot_semantic_depth_differences(
    sem_differences, 
    depth_differences, 
    bins='auto', 
    bin_count=None, 
    output_path="semantic_depth_differences.png"
):
    """
    Plots separate histograms for semantic and depth differences without combining them into a single DataFrame.
    
    Parameters:
    - sem_differences (list or array-like): Semantic difference values.
    - depth_differences (list or array-like): Depth difference values.
    - bins (int, sequence, or str, optional): Number of bins or binning strategy for the histogram.
    - bin_count (int, optional): Overrides the `bins` parameter if provided.
    - output_path (str, optional): Path to save the output histogram image.
    """
    
    # Override bins if bin_count is provided
    if bin_count is not None:
        bins = bin_count
    
    # Total number of samples for each category
    total_sem = len(sem_differences)
    total_depth = len(depth_differences)
    
    # Determine common bin edges based on both datasets
    all_differences = np.concatenate([sem_differences, depth_differences])
    bin_counts, bin_edges = np.histogram(all_differences, bins=bins)
    
    # Calculate histogram data for semantic and depth differences
    sem_counts, _ = np.histogram(sem_differences, bins=bin_edges)
    depth_counts, _ = np.histogram(depth_differences, bins=bin_edges)
    
    # Convert counts to percentages within each category
    sem_percentages = (sem_counts / total_sem) * 100 if total_sem > 0 else np.zeros_like(sem_counts)
    depth_percentages = (depth_counts / total_depth) * 100 if total_depth > 0 else np.zeros_like(depth_counts)
    
    # Initialize the plot
    plt.figure(figsize=(15, 8))  # Increased width for better visibility
    
    # Plot Semantic Differences Histogram
    plt.hist(
        sem_differences, 
        bins=bin_edges, 
        alpha=0.6, 
        label='Semantic', 
        color="#FF9999", 
        edgecolor='black', 
        weights=np.ones(len(sem_differences)) / total_sem * 100  # Normalize to percentage
    )
    
    # Plot Depth Differences Histogram
    plt.hist(
        depth_differences, 
        bins=bin_edges, 
        alpha=0.6, 
        label='Depth', 
        color="#99FF99", 
        edgecolor='black', 
        weights=np.ones(len(depth_differences)) / total_depth * 100  # Normalize to percentage
    )
    
    # Set labels and title
    plt.xlabel("Difference Value")
    plt.ylabel("Percentage within Category (%)")
    plt.title("Histogram of Semantic and Depth Differences")
    
    # Format Y-axis to show percentage
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    
    # Position the legend
    plt.legend(title="Difference Type", loc='upper right')
    
    # Calculate bin centers for annotation placement
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]
    
    # Define positions for semantic and depth bars
    sem_offsets = bin_centers - bin_width / 4
    depth_offsets = bin_centers + bin_width / 4
    
    # Annotate Semantic Differences with percentage
    for i in range(len(sem_counts)):
        if sem_counts[i] == 0:
            continue  # Skip empty bins
        plt.text(
            sem_offsets[i],
            sem_percentages[i] + 0.5,  # Slight offset above the bar
            f"{sem_percentages[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#FF0000"
        )
    
    # Annotate Depth Differences with percentage
    for i in range(len(depth_counts)):
        if depth_counts[i] == 0:
            continue  # Skip empty bins
        plt.text(
            depth_offsets[i],
            depth_percentages[i] + 0.5,  # Slight offset above the bar
            f"{depth_percentages[i]:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#006600"
        )
    
    # Adjust y-axis limit dynamically based on max percentage
    max_percentage = max(
        sem_percentages.max() if total_sem > 0 else 0,
        depth_percentages.max() if total_depth > 0 else 0
    )
    plt.ylim(0, min(max_percentage * 1.2, 100))
    
    # Configure X-axis to have ticks from 0 to max bin edge, rotated by 45 degrees
    plt.xticks(ticks=np.linspace(bin_edges[0], bin_edges[-1], num=21), rotation=45)
    
    # Adjust layout to make space for annotations and rotated labels
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Semantic and Depth differences histogram saved to {output_path}")
    plt.close()

    
def process_batches(loader):
    """
    Process all batches in the DataLoader and collect required statistics.

    Args:
        loader (DataLoader): DataLoader object.

    Returns:
        tuple: (best_index_counts, distances_to_k1, score_differences,
                total_semantic_diffs, total_depth_diffs)
    """
    best_index_counts = Counter()
    distances_to_k0 = []
    score_differences = []
    depth_differences = []
    sem_differences = []

    for batch_idx, batch in enumerate(loader):
        best_idx = batch["best_index"][0].item() - 1  
        best_index_counts[best_idx] += 1
        gt_position = np.array(batch["gt_location"][0])

        if best_idx != 0:
            k_positions = batch["k_positions"]
            if len(k_positions) > 0:
                k0 = np.array(k_positions[0][0])
                k_best = np.array(k_positions[best_idx][0])
                distance_k0_to_gt = np.linalg.norm(k0[:2] - gt_position[:2])
                distance_best_to_gt = np.linalg.norm(k_best[:2] - gt_position[:2])
                if distance_best_to_gt > 1:
                    continue
                distance = distance_k0_to_gt - distance_best_to_gt
                distances_to_k0.append(distance)

            k_scores = [score_tensor.item() for score_tensor in batch["k_scores"]]
            if len(k_scores) > 1:
                top_pick_score = k_scores[best_idx]
                k0_score = k_scores[0]
                score_diff = k0_score - top_pick_score
                score_differences.append(score_diff)

            # Compare semantic and depth rays
            metadata_path = batch["metadata_path"][0]
            try:
                with open(metadata_path, 'r') as meta_file:
                    metadata = AttrDict(yaml.safe_load(meta_file))

                # Get semantic rays for K1 and best K
                semantic_k0 = metadata.K1.semantic_rays
                best_k_key = f'K{best_idx + 1}'
                if best_k_key not in metadata:
                    print(f"Metadata key {best_k_key} not found in {metadata_path}. Skipping semantic comparison.")
                else:
                    semantic_best = metadata[best_k_key]['semantic_rays']
                    # Compare prediction_class for each ray
                    semantic_diffs = sum(
                        1 for ray1, ray2 in zip(semantic_k0, semantic_best)
                        if ray1['prediction_class'] != ray2['prediction_class']
                    )
                    sem_differences.append(semantic_diffs)

                    depth_k0 = metadata.K1.depth_rays
                    depth_best = metadata[best_k_key]['depth_rays']

                    # Compare distance_m for each ray with threshold 0.05m
                    depth_diffs = sum(
                        1 for d1, d2 in zip(depth_k0, depth_best)
                        if abs(d1['distance_m'] - d2['distance_m']) > 0.2
                    )
                    depth_differences.append(depth_diffs)

            except Exception as e:
                print(f"Error processing metadata at {metadata_path}: {e}")

        if (batch_idx + 1) % 100 == 0:
            print(f"Processed {batch_idx + 1} batches.")

    return best_index_counts, distances_to_k0, score_differences, depth_differences, sem_differences


def print_best_index_distribution(best_index_counts):
    """
    Print the distribution of best indices.

    Args:
        best_index_counts (Counter): Counter object with best index counts.
    """
    total_samples = sum(best_index_counts.values())
    print("\nDistribution of Best Index in Training Split:")
    for idx in sorted(best_index_counts):
        percentage = best_index_counts[idx] / total_samples * 100
        print(f"Best Index {idx}: {best_index_counts[idx]} samples ({percentage:.1f}%)")


def parse_arguments():
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Analyze distribution of best index in training split.")
    parser.add_argument("--config", type=str, default="modules/top_k/config_train_top_k.yaml",
                        help="Path to the YAML config file.")
    # Output filenames
    parser.add_argument("--plot_output", type=str, default="best_index_distribution.png",
                        help="Filename for best index distribution plot.")
    parser.add_argument("--distance_plot_output", type=str, default="distance_histogram.png",
                        help="Filename for distance histogram plot.")
    parser.add_argument("--score_diff_plot_output", type=str, default="score_difference_histogram.png",
                        help="Filename for score difference histogram plot.")
    parser.add_argument("--semantic_depth_plot_output", type=str, default="semantic_depth_differences.png",
                        help="Filename for semantic and depth differences plot.")
    return parser.parse_args()


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Define base output folder
    base_out_folder = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/modules/top_k/distribution_plots"
    output_paths = setup_output_paths(base_out_folder, args)

    # Load configuration and dataset split
    config = load_config(args.config)
    split = load_split(config.split_file)

    # Process scene names
    train_scenes = process_scene_names(split.test)[:-1]  # Exclude the last scene if needed
    print(f"Total training scenes: {len(train_scenes)}")

    # Construct the dataset
    dataset = TopKDataset(
        scene_names=train_scenes,
        image_base_dir=os.path.join(config.dataset_path, config.dataset),
        top_k_dir=config.top_k_results_dir,
        poses_filename=config.poses_filename
    )
    print(f"Total valid samples in training dataset: {len(dataset)}")

    # Initialize DataLoader
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    # Process all batches and collect statistics
    best_index_counts, distances_to_k0, score_differences, depth_differences, sem_differences = process_batches(loader)

    # Print best index distribution
    print_best_index_distribution(best_index_counts)

    # Generate and save plots
    plot_best_index_distribution(best_index_counts, output_paths["best_index_distribution"])
    
    if distances_to_k0:
        # Define fixed bin edges from the minimum to just above the maximum distance with a step of 0.5 meters
        bins_distance = np.arange(min(distances_to_k0), max(distances_to_k0) + 0.1, 0.5)
        
        # Call the modified plot_distance_histogram function using keyword arguments
        plot_distance_histogram(
            distances=distances_to_k0,       # Pass the distance differences
            bins=bins_distance,              # Pass the fixed bin edges
            bin_count=None,                  # Optional: Not needed since bins are explicitly defined
            output_path=output_paths["distance_histogram"]  # Specify the output path
        )

    if score_differences:
        min_diff = min(score_differences)
        max_diff = max(score_differences)
        bins_score = np.linspace(min_diff, max_diff, num=20)
        plot_score_difference_histogram(score_differences, bins_score, output_paths["score_difference_histogram"])

    if depth_differences and sem_differences:
        # Define common bin count or bin edges
        bin_count = 40          
        plot_semantic_depth_differences(
            sem_differences=sem_differences,
            depth_differences=depth_differences,
            bins='auto',          # Can be 'auto' or specific bin edges
            bin_count=bin_count,  # Set the desired number of bins
            output_path=output_paths["semantic_depth_differences"]
        )
    else:
        print("No semantic or depth differences to plot.")


if __name__ == '__main__':
    main()
