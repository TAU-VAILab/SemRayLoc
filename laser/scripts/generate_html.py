import os
import argparse
import numpy as np
from prettytable import PrettyTable
import matplotlib.pyplot as plt

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--result_dir", type=str, required=True)
opt = parser.parse_args()
result_dir = opt.result_dir

def html_img_link(img_path):
    return rf'<img src="{img_path}" width=400px>'

# Load filenames
result_fns = sorted(os.listdir(os.path.join(result_dir, "results")))
score_map_fns = sorted(os.listdir(os.path.join(result_dir, "score_maps")))
rot_map_fns = sorted(os.listdir(os.path.join(result_dir, "rot_maps")))
query_image_fns = sorted(os.listdir(os.path.join(result_dir, "query_images")))
terr_fns = sorted(os.listdir(os.path.join(result_dir, "terrs")))
rerr_fns = sorted(os.listdir(os.path.join(result_dir, "rerrs")))

# Load error values
terrs = []
rerrs = []
for terr_fn in terr_fns:
    terrs.append(np.loadtxt(os.path.join(result_dir, "terrs", terr_fn)).reshape(-1))
for rerr_fn in rerr_fns:
    rerrs.append(np.loadtxt(os.path.join(result_dir, "rerrs", rerr_fn)).reshape(-1))
terrs = np.concatenate(terrs, axis=0)
rerrs = np.concatenate(rerrs, axis=0)

# Ensure all lists have the same length
print(len(result_fns))
print(len(score_map_fns))
print(len(rot_map_fns))
print(len(query_image_fns))
print(len(terrs))
print(len(rerrs))

assert (
    len(result_fns)
    == len(score_map_fns)
    == len(rot_map_fns)
    == len(query_image_fns)
    == len(rerrs)
    == len(terrs)
), "Mismatch in the number of files loaded."

n_results = len(terrs)
print(f"{n_results} results loaded.")

# Generate HTML report (existing functionality)
for rng in [(0, 999)]:  # You can modify this as needed
    n_rows = 0
    table = PrettyTable()
    table.field_names = [
        "idx",
        "trans_err(m)",
        "rot_err(degree)",
        "result",
        "score_map",
        "rot_map",
        "query_image",
    ]
    for i in range(n_results):
        if terrs[i] < rng[0] or terrs[i] > rng[1]:  # change condition here
            continue
        n_rows += 1
        result_path = os.path.join(".", "results", result_fns[i])
        score_map_path = os.path.join(".", "score_maps", score_map_fns[i])
        rot_map_path = os.path.join(".", "rot_maps", rot_map_fns[i])
        query_image_path = os.path.join(".", "query_images", query_image_fns[i])
        table.add_row(
            [
                i,
                f"{terrs[i]:.4f}",
                f"{rerrs[i]:.4f}",
                html_img_link(result_path),
                html_img_link(score_map_path),
                html_img_link(rot_map_path),
                html_img_link(query_image_path),
            ]
        )

    html_str = table.get_html_string()
    html_str = html_str.replace(r"&lt;", r"<")
    html_str = html_str.replace(r"&gt;", r">")
    html_str = html_str.replace(r"&quot;", r'"')

    with open(
        os.path.join(result_dir, f"results--{rng[0]:.1f}-{rng[1]:.1f}.html"), "w"
    ) as f:
        f.write(html_str)

# Plot CDF for Translation Errors
pdf, bins = np.histogram(
    np.clip(terrs, a_min=None, a_max=1.05), bins=21, range=(0, 1.05), density=True
)
cdf = np.cumsum(pdf * np.diff(bins))  # Correct CDF calculation
rng_cdf = bins[1:]  # Upper edges of bins

# Plotting the CDF
plt.figure(figsize=(8, 6))
plt.plot(rng_cdf, cdf, marker='o')
plt.xlabel('Translation Error (m)')
plt.ylabel('Cumulative Probability')
plt.title('CDF of Translation Errors')
plt.grid(True)
plt.savefig(os.path.join(result_dir, "cdf_translation_errors.png"))
plt.show()

# Compute Recall Rates based on Translation Errors
trans_thresholds = [0.1, 0.5, 1.0]  # in meters
recall_rates = {}
for thresh in trans_thresholds:
    recall = np.sum(terrs <= thresh) / len(terrs)
    recall_rates[thresh] = recall

# Display Recall Rates in a Table
recall_table = PrettyTable()
recall_table.field_names = ["Threshold (m)", "Recall (%)"]
for thresh, recall in recall_rates.items():
    recall_table.add_row([f"≤ {thresh:.1f}", f"{recall * 100:.2f}"])

print("\nRecall Rates based on Translation Errors:")
print(recall_table)

# === Added Section: Compute Combined Recall Rate ===

# Define combined thresholds
combined_trans_threshold = 1.0  # in meters
combined_rot_threshold = 30.0   # in degrees

# Calculate combined recall rate: Translation ≤ 1.0 m AND Rotation ≤ 30°
combined_recall = np.sum((terrs <= combined_trans_threshold) & (rerrs <= combined_rot_threshold)) / len(terrs)

# Create a new table for combined recall rates
combined_recall_table = PrettyTable()
combined_recall_table.field_names = ["Translation Threshold (m)", "Rotation Threshold (°)", "Combined Recall (%)"]

# Add the combined recall rate to the table
combined_recall_table.add_row([f"≤ {combined_trans_threshold:.1f}", f"≤ {combined_rot_threshold:.0f}", f"{combined_recall * 100:.2f}"])

# Print the combined recall rates table
print("\nCombined Recall Rates (Translation ≤ 1.0 m AND Rotation ≤ 30°):")
print(combined_recall_table)

# (Optional) Save the Combined Recall Rate to a text file
with open(os.path.join(result_dir, 'combined_recall_rate.txt'), 'w') as f:
    f.write("Translation Threshold (m)\tRotation Threshold (°)\tCombined Recall (%)\n")
    f.write(f"≤ {combined_trans_threshold:.1f}\t≤ {combined_rot_threshold:.0f}\t{combined_recall * 100:.2f}\n")

# === End of Added Section ===

# (Optional) Save CDF data and recall rates
np.savetxt(
    os.path.join(result_dir, 'cdf.txt'),
    np.stack([rng_cdf, cdf], axis=1),
    fmt='%.4f',
    header='Error(m)\tCDF',
    comments=''
)

with open(os.path.join(result_dir, 'recall_rates.txt'), 'w') as f:
    f.write("Threshold(m)\tRecall(%)\n")
    for thresh, recall in recall_rates.items():
        f.write(f"≤ {thresh:.1f}\t{recall * 100:.2f}\n")

# (Optional) Save Combined Recall Rates
with open(os.path.join(result_dir, 'combined_recall_rates.txt'), 'w') as f:
    f.write("Translation Threshold (m)\tRotation Threshold (°)\tCombined Recall (%)\n")
    f.write(f"≤ {combined_trans_threshold:.1f}\t≤ {combined_rot_threshold:.0f}\t{combined_recall * 100:.2f}\n")
