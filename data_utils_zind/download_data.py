import os
import requests
from tqdm import tqdm
import zipfile

# List of URLs to download
urls = [
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_00.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_01.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_02.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_03.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_04.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_05.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_06.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_07.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_08.zip",
    # # Skipping the corrupted file
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_10.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_11.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_12.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_13.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_14.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_15.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_16.zip",
    # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_17.zip",
]

# Directory to save the files
save_dir = "/datadrive2/CRM.AI.Research/TeamFolders/Email/repo_yuval/FloorPlan/Semantic_Floor_plan_localization/data/structured3d_perspective_full/"

# Ensure the directory exists
os.makedirs(save_dir, exist_ok=True)

# Directory to extract the files
extract_dir = os.path.join(save_dir, "Structured3D")

# Ensure the extract directory exists
os.makedirs(extract_dir, exist_ok=True)

# Download and unzip each file
for url in urls:
    file_name = os.path.join(save_dir, os.path.basename(url))
    print(f"Downloading {url} to {file_name}...")

    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KiB

    if response.status_code == 200:
        with open(file_name, 'wb') as file, tqdm(
            desc=file_name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        print(f"Successfully downloaded {file_name}")

        # Unzip the file and move contents from inner Structured3D folder
        print(f"Unzipping {file_name} into {extract_dir}...")
        try:
            with zipfile.ZipFile(file_name, 'r') as zip_ref:
                # Extract the contents to a temporary directory
                temp_dir = os.path.join(save_dir, "temp")
                os.makedirs(temp_dir, exist_ok=True)
                zip_ref.extractall(temp_dir)

                # Move the contents of the inner Structured3D folder to the final directory
                inner_dir = os.path.join(temp_dir, "Structured3D")
                for item in os.listdir(inner_dir):
                    s = os.path.join(inner_dir, item)
                    d = os.path.join(extract_dir, item)
                    if os.path.isdir(s):
                        os.rename(s, d)
                    else:
                        os.rename(s, d)

                # Clean up the temporary directory
                os.rmdir(inner_dir)
                os.rmdir(temp_dir)

            print(f"Successfully unzipped and moved contents of {file_name}")
        except zipfile.BadZipFile:
            print(f"Error: {file_name} is a corrupt ZIP file.")
    else:
        print(f"Failed to download {url}")

print("Download and extraction complete!")