import os
import tqdm
import cv2

data_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/test_data_set_full/structured3d_perspective_full"
# get all scene directories in the path
scenes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and d.startswith('scene_')] 
scenes.sort(key=lambda x: int(x.split('_')[-1]))
# scenes = sorted(os.listdir(data_dir))
# scenes = scenes[1447:1450]
# scenes = scenes[0:500] #1
# scenes = scenes[500:1000]#2
# scenes = scenes[1000:1500]#3
# scenes = scenes[1500:2000]#4
# scenes = scenes[2000:2500]#5
# scenes = scenes[2500:3000]#6
# scenes = scenes[3000:3500]#7

for scene in tqdm.tqdm(scenes):
    print(scene)
    # path to the 'rgb' directory within each scene
    img_dir = os.path.join(data_dir, scene, "rgb")
    
    if not os.path.exists(img_dir):
        continue  # skip if the 'rgb' directory does not exist

    for img in sorted(os.listdir(img_dir)):
        img_path = os.path.join(img_dir, img)
        n_img = img[:-4]  # remove the file extension
        
        # Load the image
        im = cv2.imread(img_path)
        
        # Check if the image was loaded correctly
        if im is None:
            continue  # skip if the image could not be loaded
        
        # Check if the image resolution is already 640x360
        if im.shape[1] == 640 and im.shape[0] == 360:
            continue  # skip resizing if the resolution is already 640x360

        # Resize the image to 640x360 if it is not already that size
        im_resized = cv2.resize(im, (640, 360))
        
        # Save the resized image as a .png file
        cv2.imwrite(os.path.join(img_dir, f"{n_img}.png"), im_resized)
        
        # Optionally remove the original image (uncomment if needed)
        # os.remove(img_path)

print("Processing complete.")
