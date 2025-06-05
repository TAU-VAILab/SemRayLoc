import os
import tqdm
import cv2
import numpy as np

# pano dimensions
PANO_W, PANO_H = 1024, 512
FOV_X = 80.0

data_dir = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/structured3d_perspective/test_data_set_full"
scenes = [d for d in os.listdir(data_dir)
          if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('scene_')]
scenes.sort(key=lambda x: int(x.split('_')[-1]))
scenes = scenes[1190:]

for scene in tqdm.tqdm(scenes):
    img_dir       = os.path.join(data_dir, scene, "rgb")
    #continu if not exists
    if not os.path.exists(img_dir):
        print(f"Image directory {img_dir} does not exist. Skipping...")
        continue
    out_dir       = os.path.join(data_dir, scene, "rgb_low_res")
    os.makedirs(out_dir, exist_ok=True)

    for img_name in sorted(os.listdir(img_dir)):
        im = cv2.imread(os.path.join(img_dir, img_name))
        if im is None: continue

        h0, w0 = im.shape[:2]     # 360, 640

        # 1) compute the “true” number of pixels that an 80°×45° window
        #    would cover on the original 1028×512 pano:
        #      low_w ≃ 1024 * (80/360) ≃ 228 px
        #      low_h ≃ 512  * (45/180) ≃ 128 px   (45° vert fov = 360/640*80)
        low_w = int(round(PANO_W * (FOV_X / 360.0)))
        vert_fov = (h0 / w0) * FOV_X           # = 360/640 * 80 = 45°
        low_h = int(round(PANO_H * (vert_fov / 180.0)))

        # 2) down‐sample your 640×360 image to that “spherical” resolution:
        im_low = cv2.resize(im,
                            (low_w, low_h),
                            interpolation=cv2.INTER_CUBIC)

        # 3) up‐sample back to a 512-wide input (preserving 16∶9):
        #      new_h = 512 * (360/640) = 288
        up_w = 512
        up_h = int(round(up_w * (h0 / w0)))
        im_up = cv2.resize(im_low,
                           (up_w, up_h),
                           interpolation=cv2.INTER_CUBIC)

        # save
        fn, _ = os.path.splitext(img_name)
        cv2.imwrite(os.path.join(out_dir, f"{fn}.png"), im_up)
