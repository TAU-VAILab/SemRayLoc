# Semantic Floor Plan Localization

This repository contains code for semantic-aware floorplan localization using depth and semantic cues. The framework leverages deep learning to predict depth and semantic rays, enabling accurate camera pose estimation in indoor environments.

## Setup

1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Data Preparation

1. **Download Datasets**
   - Use the provided helper functions/scripts to download the S3D and ZInD datasets.
   - Place the datasets in the `Data/` directory as follows:
     - `Data/S3D/`
     - `Data/zind/`

2. **Resize Images**
   - Use the helper script to resize images for training:
   ```bash
   python data_utils/resize_images.py
   ```
   - This will resize all images in the dataset to the required input size.

3. **Generate Raycast Maps**
   - Use the following script to generate grid raycasts for floorplan maps:
   ```bash
   python -m data_utils.generate_maps_grid_raycasts_multi_thread
   ```
   - This will create the necessary depth and semantic raycast maps for each scene.

4. **Map Room Types**
   - Use the helper functions in `modules/semantic/semantic_mapper.py` to map room types as needed for your dataset.

## Training

- Configure your training parameters in the YAML files under `Train_models/configurations/S3D/`.
- Start training with:
  ```bash
  python -m Train_models.Train
  ```

## Evaluation

- Evaluate localization performance with:
  ```bash
  python -m evaluation.eval_localization
  ```

## Pretrained Weights

- Download pretrained weights and place them in the appropriate directory (e.g., `modules/weights/s3d/`).
- [Download weights here](<PLACEHOLDER_FOR_LINK>)

## Requirements

See `requirements.txt` for all dependencies.

## Citation

If you use this code, please cite our work (citation coming soon). 