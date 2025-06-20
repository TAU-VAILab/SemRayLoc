# Semantic Floor Plan Localization

[![GitHub](https://img.shields.io/badge/GitHub-Project-blue?logo=github)](https://github.com/your-repo)
[Paper (arXiv)](https://arxiv.org/abs/your-arxiv-id)

This is the official code repository for our work on semantic-aware floorplan localization. The framework leverages both semantic and depth information for accurate camera pose estimation in indoor environments.

---

## TL;DR
We introduce a novel floorplan localization framework that leverages both semantic and depth information. Our approach uses semantic rays to capture important structural elements like windows and doors, combined with depth cues for accurate camera pose estimation. The method employs a coarse-to-fine strategy, first sampling a small set of rays for initial localization, then refining in high-probability regions. Our experiments show significant improvements over state-of-the-art methods, with the added benefit of easily incorporating room labels for enhanced accuracy.

## Abstract
Floorplans provide a compact representation of the building's structure, revealing not only layout information but also detailed semantics such as the locations of windows and doors. However, contemporary floorplan localization techniques mostly focus on matching depth-based structural cues, ignoring the rich semantics communicated within floorplans. In this work, we introduce a semantic-aware localization framework that jointly estimates depth and semantic rays, consolidating over both for predicting a structural-semantic probability volume. Our probability volume is constructed in a coarse-to-fine manner: We first sample a small set of rays to obtain an initial low-resolution probability volume. We then refine these probabilities by performing a denser sampling only in high-probability regions and process the refined values for predicting a 2D location and orientation angle. We conduct an evaluation on two standard floorplan localization benchmarks. Our experiments demonstrate that our approach substantially outperforms state-of-the-art methods, achieving significant improvements in recall metrics compared to prior works. Moreover, we demonstrate that our framework can easily incorporate additional metadata such as room labels, enabling additional gains in both accuracy and efficiency.

---

## Links
- [GitHub Project Page](https://github.com/your-repo)
- [Paper (arXiv)](https://arxiv.org/abs/your-arxiv-id)
- [Structured3D Dataset](https://structured3d-dataset.org/)
- [ZInD Dataset](https://zind.cs.princeton.edu/)

---

## Data Preparation Pipeline

For each dataset (S3D, ZInD):

1. **Download the dataset**
   - Use the official dataset pages:
     - [Structured3D Download](https://structured3d-dataset.org/)
     - [ZInD Download](https://zind.cs.princeton.edu/)
   - You can use the provided script to automate download and extraction for S3D:
     ```bash
     python data_utils/s3d/download_and_extract.py
     ```

2. **Create Processed Datasets**
   - After downloading, process the raw data to create a processed folder for each dataset:
     ```bash
     # For S3D
     python data_utils/s3d/create_data_sets.py
     # For ZInD (adapt as needed for ZInD structure)
     python data_utils/zind/create_data_sets.py
     ```
   - This will create a `processed` folder with the required structure.

3. **Resize Images**
   - Resize all images to the required input size:
     ```bash
     python data_utils/resize_images.py
     ```

4. **Generate Raycast Maps**
   - Generate grid raycasts for floorplan maps:
     ```bash
     python -m data_utils.generate_maps_grid_raycasts_multi_thread
     ```

5. **Map Room Types**
   - Use the helper functions in `modules/semantic/semantic_mapper.py` to map room types as needed for your dataset.

---

## Training
- Each training run is controlled by a config file (YAML) in `Train_models/configurations/S3D/` or the corresponding dataset folder.
- Adjust the main parameters (e.g., learning rate, batch size, loss weights, etc.) in the config file before training.
- Start training:
  ```bash
  python -m Train_models.Train
  ```

## Evaluation
- For evaluation, specify the evaluation config, weights directory, and results directory in the evaluation script/config.
- Run evaluation:
  ```bash
  python -m evaluation.eval_localization
  ```

---

## Pretrained Weights
- Download pretrained weights and place them in the appropriate directory (e.g., `modules/weights/s3d/`).
- [Download weights here](<PLACEHOLDER_FOR_LINK>)

---

## Requirements
See `requirements.txt` for all dependencies.

---

## Citation
If you use this code, please cite our work (citation coming soon). 