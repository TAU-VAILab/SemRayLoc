import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
from attrdict import AttrDict

from modules.mono.depth_net_pl import depth_net_pl
from modules.semantic.semantic_net_pl import semantic_net_pl
from data_utils import GridSeqDataset
from utils.localization_utils import localize, get_ray_from_depth, get_ray_from_semantics

def compute_and_save_prob_vols(dataset_dir, desdf_path, prob_vols_dir, start_scene, end_scene, depth_net, semantic_net, device):
    # Initialize the dataset with the specified scene range
    test_set = GridSeqDataset(
        dataset_dir=dataset_dir,
        scene_names=None,  # We'll generate scene names based on the range
        L=0,
        start_scene=start_scene,
        end_scene=end_scene,
    )

    # Preload desdf and semantic data for each scene
    desdfs = {}
    semantics = {}
    scenes_processed = set()

    for idx in tqdm(range(len(test_set)), desc="Processing images"):
        data = test_set[idx]
        scene_idx = np.searchsorted(test_set.scene_start_idx, idx, side='right') - 1
        scene = test_set.scene_names[scene_idx]
        scene_number = int(scene.split('_')[1])
        scene_name = f"scene_{scene_number}"
        idx_within_scene = idx - test_set.scene_start_idx[scene_idx]

        # Load desdf and semantic data if not already loaded
        if scene_name not in scenes_processed:
            desdf_file = os.path.join(desdf_path, scene_name, "desdf.npy")
            semantic_file = os.path.join(desdf_path, scene_name, "color.npy")

            if not os.path.exists(desdf_file) or not os.path.exists(semantic_file):
                print(f"Desdf or semantic file missing for {scene_name}. Skipping scene.")
                continue

            desdfs[scene_name] = np.load(desdf_file, allow_pickle=True).item()
            semantics[scene_name] = np.load(semantic_file, allow_pickle=True).item()
            scenes_processed.add(scene_name)

        desdf = desdfs[scene_name]
        semantic = semantics[scene_name]
        ref_img_torch = torch.tensor(data["ref_img"], device=device).unsqueeze(0)

        # Compute depth rays
        with torch.no_grad():
            pred_depths, _, _ = depth_net.encoder(ref_img_torch, None)
            pred_depths = pred_depths.squeeze(0).cpu().numpy()
            pred_rays_depth = get_ray_from_depth(pred_depths)

        # Compute semantic rays
        with torch.no_grad():
            _, _, prob = semantic_net.encoder(ref_img_torch, None)
            prob_squeezed = prob.squeeze(dim=0)
            sampled_indices = torch.multinomial(prob_squeezed, num_samples=1, replacement=True)
            sampled_indices = sampled_indices.squeeze(dim=1)
            sampled_indices_np = sampled_indices.cpu().numpy()
            pred_rays_semantic = get_ray_from_semantics(sampled_indices_np)

        # Compute prob_vols
        with torch.no_grad():
            prob_vol_pred_depth, _, _, _ = localize(
                torch.tensor(desdf["desdf"]), torch.tensor(pred_rays_depth, device="cpu"), return_np=False
            )
            prob_vol_pred_semantic, _, _, _ = localize(
                torch.tensor(semantic["desdf"]), torch.tensor(pred_rays_semantic, device="cpu"), return_np=False
            )

        # Save prob_vols
        prob_vol_scene_dir = os.path.join(prob_vols_dir, scene_name)
        os.makedirs(prob_vol_scene_dir, exist_ok=True)

        np.save(os.path.join(prob_vol_scene_dir, f"depth_prob_vol_{idx_within_scene}.npy"), prob_vol_pred_depth.cpu().numpy())
        np.save(os.path.join(prob_vol_scene_dir, f"semantic_prob_vol_{idx_within_scene}.npy"), prob_vol_pred_semantic.cpu().numpy())

        # Save the ground truth pose for later use
        gt_pose = data['ref_pose']
        np.save(os.path.join(prob_vol_scene_dir, f"gt_pose_{idx_within_scene}.npy"), gt_pose)

if __name__ == "__main__":
    # Paths and parameters
    dataset_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full"
    desdf_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/desdf"
    prob_vols_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/prob_vols"
    start_scene = 0
    end_scene = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pretrained models
    depth_checkpoint_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/logs/full/21_9_fix_depth_to_match_paper/depth/depth_net-epoch=94-loss-valid=0.47.ckpt"
    semantic_checkpoint_path = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/logs/full/21_9_fix_depth_to_match_paper/semantic/semantic_net-epoch=10-loss-valid=0.26-v1.ckpt"

    depth_net = depth_net_pl.load_from_checkpoint(
        checkpoint_path=depth_checkpoint_path,
        d_min=0.1,
        d_max=15.0,
        d_hyp=-0.2,
        D=128,
    ).to(device)
    depth_net.eval()

    semantic_net = semantic_net_pl.load_from_checkpoint(
        checkpoint_path=semantic_checkpoint_path,
        num_classes=4,
    ).to(device)
    semantic_net.eval()

    # Compute and save prob_vols
    compute_and_save_prob_vols(
        dataset_dir=dataset_dir,
        desdf_path=desdf_path,
        prob_vols_dir=prob_vols_dir,
        start_scene=start_scene,
        end_scene=end_scene,
        depth_net=depth_net,
        semantic_net=semantic_net,
        device=device
    )
