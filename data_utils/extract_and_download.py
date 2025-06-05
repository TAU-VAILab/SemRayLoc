import os
import requests
from tqdm import tqdm
import zipfile
import shutil

def download_and_extract(url, save_dir, extract_dir):
    """
    Downloads the zip from the given URL (if not already downloaded)
    and extracts its contents. Extraction is done by unzipping into a temporary
    directory. It then moves the items inside the inner 'Structured3D' folder into
    the target extract_dir if they are not already present.
    """
    zip_filename = os.path.basename(url)
    zip_filepath = os.path.join(save_dir, zip_filename)
    
    # Download if the file doesn't exist already
    # if not os.path.exists(zip_filepath):
    print(f"Downloading {url} to {zip_filepath}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024  # 1 KiB

    if response.status_code == 200:
        with open(zip_filepath, 'wb') as file, tqdm(
            desc=zip_filename,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        print(f"Successfully downloaded {zip_filepath}")
    else:
        print(f"Failed to download {url}")
        return
    # else:
    #     print(f"Zip file {zip_filepath} already exists. Skipping download.")

    # Extraction: unzip to a temporary folder first
    print(f"Extracting {zip_filepath} into temporary directory...")
    try:
        with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
            temp_dir = os.path.join(save_dir, "temp")
            os.makedirs(temp_dir, exist_ok=True)
            # Extract each member individually to handle corrupted files gracefully
            for member in zip_ref.namelist():
                try:
                    zip_ref.extract(member, temp_dir)
                except Exception as e:
                    print(f"Skipping corrupted file {member}: {e}")
                    continue

            # Move the files from the inner directory "Structured3D" if it exists
            inner_dir = os.path.join(temp_dir, "Structured3D")
            if os.path.exists(inner_dir):
                # For each item (scene folder or file) inside "Structured3D"
                for item in os.listdir(inner_dir):
                    src_item = os.path.join(inner_dir, item)
                    dest_item = os.path.join(extract_dir, item)
                    if os.path.exists(dest_item):
                        print(f"Item {dest_item} already exists. Skipping.")
                    else:
                        print(f"Moving {src_item} to {dest_item}")
                        shutil.move(src_item, dest_item)
                # Clean up temporary directory
                shutil.rmtree(temp_dir)
            else:
                print("Expected inner directory 'Structured3D' not found.")
            print(f"Extraction complete for {zip_filepath}")
    except zipfile.BadZipFile:
        print(f"Error: {zip_filepath} is a corrupt ZIP file.")
    except Exception as e:
        print(f"An error occurred while extracting {zip_filepath}: {e}")

def cleanup_dataset(extract_dir):
    """
    For every scene in the dataset:
      - In each 'full' folder, delete all PNG files except 'rgb_rawlight.png'.
      - Remove directories named 'empty' or 'simple'.
    """
    # Step 1: Process all "full" directories
    for root, dirs, files in os.walk(extract_dir):
        if os.path.basename(root) == "full":
            for file in files:
                if file.endswith(".png") and file != "rgb_rawlight.png":
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")

    # Step 2: Remove directories named "empty" or "simple"
    for root, dirs, files in os.walk(extract_dir, topdown=False):
        for d in dirs:
            if d in ["empty", "simple"]:
                dir_to_delete = os.path.join(root, d)
                try:
                    shutil.rmtree(dir_to_delete)
                    print(f"Deleted directory: {dir_to_delete}")
                except Exception as e:
                    print(f"Error deleting directory {dir_to_delete}: {e}")

def main():
    # List of URLs to download
    urls = [
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_00.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_01.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_02.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_03.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_04.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_05.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_06.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_07.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_08.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_09.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_10.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_11.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_12.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_13.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_14.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_15.zip",
        # "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_16.zip",
        "https://zju-kjl-jointlab-azure.kujiale.com/zju-kjl-jointlab/Structured3D/Structured3D_panorama_17.zip"
    ]

    # Define directories for saving zips and extraction
    save_dir = "/home/yuvalg/projects/Semantic_Floor_plan_localization/data/structured3d"
    extract_dir = os.path.join(save_dir, "structured3d_panorama")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)

    # Process each zip URL
    for url in urls:
        download_and_extract(url, save_dir, extract_dir)

    # After download and extraction, perform cleanup on the dataset
    print("Starting cleanup of extracted files...")
    cleanup_dataset(extract_dir)
    print("Download, extraction, and cleanup complete!")

if __name__ == "__main__":
    main()
