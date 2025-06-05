# utils/result_utils.py

import os
import numpy as np
import subprocess
from PIL import Image

def save_acc_and_orn_records(acc_record, acc_orn_record, weight_dir):
    os.makedirs(weight_dir, exist_ok=True)
    if acc_record is not None:
        np.save(os.path.join(weight_dir, "acc_record.npy"), acc_record)
    if acc_orn_record is not None:
        np.save(os.path.join(weight_dir, "acc_orn_record.npy"), acc_orn_record)

def calculate_recalls(acc_record, acc_orn_record):
    recalls = {        
        "1m": np.sum(acc_record < 1) / acc_record.shape[0],
        "0.5m": np.sum(acc_record < 0.5) / acc_record.shape[0],
        "0.1m": np.sum(acc_record < 0.1) / acc_record.shape[0],
        "1m 30 deg": np.sum(np.logical_and(acc_record < 1, acc_orn_record < 30)) / acc_record.shape[0] if acc_orn_record is not None else 0 ,
    }
    return recalls

def save_recalls(recalls, weight_dir, combination_name):
    with open(os.path.join(weight_dir, f"recalls_{combination_name}.txt"), "w") as f:
        for key, value in recalls.items():
            f.write(f"{key} recall = {value}\n")

import os
import subprocess
from PIL import Image


def create_combined_results_table(combined_recalls, results_dir):
    table_file = os.path.join(results_dir, "combined_results_table.txt")
    latex_table_file = os.path.join(results_dir, "combined_results_table.tex")
    png_table_file = os.path.join(results_dir, "combined_results_table.png")

    # Collect recall keys
    recall_keys = ["0.1m","0.5m","1m","1m 30 deg"]

    # Prepare a dictionary to store the best score for each recall type
    best_scores = {key: -1 for key in recall_keys}
    best_combinations = {key: None for key in recall_keys}

    # First, determine the best (highest) recall score for each recall type
    for stage, recalls in combined_recalls.items():
        # Update recalls from 0-1 to 0-100 scale
        for recall_type, value in recalls.items():
            recalls[recall_type] = value * 100
            if value > best_scores[recall_type]:
                best_scores[recall_type] = value
                best_combinations[recall_type] = stage

    # Generate the text-based table
    with open(table_file, "w") as f:
        f.write("Stage\t0.1m\t0.5m\t1m\t1m 30 deg\n")
        for stage, recalls in combined_recalls.items():
            f.write(f"{stage}\t")
            f.write("\t".join([f"{recalls[key]:.2f}" for key in recall_keys]))
            f.write("\n")

    # Generate the LaTeX table
    with open(latex_table_file, "w") as f:
        f.write("\\documentclass{article}\n")
        f.write("\\usepackage[utf8]{inputenc}\n")
        f.write("\\usepackage{amsmath}\n")
        f.write("\\usepackage{graphicx}\n")
        f.write("\\begin{document}\n")
        f.write("\\begin{table}[h!]\n")
        f.write("\\centering\n")
        f.write("\\begin{tabular}{|l|" + "c|" * len(recall_keys) + "}\n")
        f.write("\\hline\n")

        # Header row
        f.write("Stage & " + " & ".join(recall_keys) + " \\\\\n")
        f.write("\\hline\n")

        # Data rows
        for stage, recalls in combined_recalls.items():
            # Escape underscores in stage names for LaTeX
            escaped_stage = stage.replace("_", "\\_")
            f.write(f"{escaped_stage}")
            f.write(" & ")

            recall_values = []
            for key in recall_keys:
                # Bold the best score
                if best_combinations[key] == stage:
                    recall_values.append(f"\\textbf{{{recalls[key]:.2f}}}")
                else:
                    recall_values.append(f"{recalls[key]:.2f}")
            f.write(" & ".join(recall_values) + " \\\\\n")
            f.write("\\hline\n")

        f.write("\\end{tabular}\n")
        f.write("\\caption{Recall comparison for conditional input, predicted output, and ground truth.}\n")
        f.write("\\end{table}\n")
        f.write("\\end{document}\n")

    print(f"LaTeX table saved to: {latex_table_file}")

    # Compile the LaTeX file to PDF and convert to PNG
    try:
        # Change directory to results_dir to handle auxiliary files
        current_dir = os.getcwd()
        os.chdir(results_dir)

        # Compile the LaTeX file
        subprocess.run(["pdflatex", latex_table_file], check=True)

        # Convert PDF to PNG
        pdf_file = os.path.splitext(latex_table_file)[0] + ".pdf"
        subprocess.run(["convert", "-density", "300", pdf_file, png_table_file], check=True)

        print(f"LaTeX table converted to PNG: {png_table_file}")

        # Optionally, display the generated PNG
        img = Image.open(png_table_file)
        img.show()

        # Change back to the original directory
        os.chdir(current_dir)

    except subprocess.CalledProcessError as e:
        print("Error during LaTeX compilation or PNG conversion:", e)

