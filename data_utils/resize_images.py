import os
import tqdm
import cv2

data_dir = r"C:\Users\t-ygrader\Desktop\UV\Personal\floor_plan_localication\Semantic_Floor_plan_localization\Data\S3D\processed"
# get all scene directories in the path
scenes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('scene_')] 
scenes.sort(key=lambda x: int(x.split('_')[-1]))
print(len(scenes))

for scene in tqdm.tqdm(scenes):
    img_dir = os.path.join(data_dir, scene, "rgb")
    
    if not os.path.exists(img_dir):
        continue

    for img in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img)
        
        im = cv2.imread(img_path)
        
        if im is None:
            continue
        
        im_resized = cv2.resize(im, (640, 360))
        # Save the resized image as a .png file
        cv2.imwrite(os.path.join(img_dir, f"{img}"), im_resized)
        
print("Processing complete.")
