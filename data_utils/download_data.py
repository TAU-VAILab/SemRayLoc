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
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_13.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_14.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_15.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_16.zip",
    "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_perspective_full_17.zip",
]

import os
import requests
from tqdm import tqdm
import zipfile
import shutil

def download_and_extract(url, save_dir, extract_dir):
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
    else:
        print(f"Failed to download {url}")
        return

    # Validate and unzip the file
    print(f"Unzipping {file_name} into {extract_dir}...")
    try:
        with zipfile.ZipFile(file_name, 'r') as zip_ref:
            # Validate the ZIP file
            bad_file = zip_ref.testzip()
            if bad_file:
                print(f"Corrupt file detected in the zip: {bad_file}")
                return
            temp_dir = os.path.join(save_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            zip_ref.extractall(temp_dir)

            inner_dir = os.path.join(temp_dir, "Structured3D")
            if os.path.exists(inner_dir):
                for item in os.listdir(inner_dir):
                    s = os.path.join(inner_dir, item)
                    d = os.path.join(extract_dir, item)
                    shutil.move(s, d)
                # Clean up temporary directories
                shutil.rmtree(temp_dir)
            else:
                print("Expected inner directory 'Structured3D' not found.")
            print(f"Successfully unzipped and moved contents of {file_name}")
    except zipfile.BadZipFile:
        print(f"Error: {file_name} is a corrupt ZIP file.")
    except Exception as e:
        print(f"An error occurred: {e}")

# # List of URLs
# urls = [
#     "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_00.zip",
#     "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_00.zip",
#     "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_01.zip",
#     "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_02.zip",
#     "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_03.zip",
#     "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_04.zip",
#     "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_05.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_06.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_07.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_08.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_09.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_10.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_11.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_12.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_13.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_14.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_15.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_16.zip",
#     # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_17.zip"
# ]

save_dir = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/structured3d"
extract_dir = os.path.join(save_dir, "structured3d_perspective")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(extract_dir, exist_ok=True)

for url in urls:
    download_and_extract(url, save_dir, extract_dir)

print("Download and extraction complete!")
