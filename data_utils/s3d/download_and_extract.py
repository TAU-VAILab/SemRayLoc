"""
Data Download Module for Structured3D Dataset

This module handles the downloading and extraction of the Structured3D dataset.
It provides functionality to download multiple files, validate them, and extract
them to the appropriate directories.
"""

import os
import requests
from tqdm import tqdm
import zipfile
import shutil
from typing import List, Optional
from pathlib import Path
import logging
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Configuration settings for data download."""
    BASE_URL = "https://structured3d-dataset.org/Structured3D_perspective_full_{:02d}.zip"
    BLOCK_SIZE = 1024  # 1 KiB
    CHUNK_SIZE = 8192  # 8 KiB for streaming
    MAX_RETRIES = 3
    RETRY_DELAY = 5  # seconds

# List of URLs to download
DOWNLOAD_URLS = [
    Config.BASE_URL.format(i) for i in range(9)  # 00 to 08
]

def create_directories(save_dir: str, extract_dir: str) -> None:
    """Create necessary directories for data storage.
    
    Args:
        save_dir: Directory to save downloaded files
        extract_dir: Directory to extract files to
    """
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    logger.info(f"Created directories: {save_dir} and {extract_dir}")

def download_file(url: str, file_path: str) -> bool:
    """Download a file with progress bar and error handling.
    
    Args:
        url: URL to download from
        file_path: Path to save the file to
        
    Returns:
        bool: True if download was successful, False otherwise
    """
    for attempt in range(Config.MAX_RETRIES):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(file_path, 'wb') as file, tqdm(
                desc=os.path.basename(file_path),
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=Config.CHUNK_SIZE):
                    if chunk:
                        bar.update(len(chunk))
                        file.write(chunk)
            
            logger.info(f"Successfully downloaded {file_path}")
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if attempt < Config.MAX_RETRIES - 1:
                logger.info(f"Retrying in {Config.RETRY_DELAY} seconds...")
                time.sleep(Config.RETRY_DELAY)
            continue
            
    logger.error(f"Failed to download {url} after {Config.MAX_RETRIES} attempts")
    return False

def extract_zip(zip_path: str, extract_dir: str) -> bool:
    """Extract and validate a ZIP file.
    
    Args:
        zip_path: Path to the ZIP file
        extract_dir: Directory to extract to
        
    Returns:
        bool: True if extraction was successful, False otherwise
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Validate the ZIP file
            bad_file = zip_ref.testzip()
            if bad_file:
                logger.error(f"Corrupt file detected in the zip: {bad_file}")
                return False
                
            # Create temporary directory for extraction
            temp_dir = os.path.join(os.path.dirname(zip_path), "temp")
            os.makedirs(temp_dir, exist_ok=True)
            
            # Extract to temporary directory
            zip_ref.extractall(temp_dir)
            
            # Move contents from temporary directory to final location
            inner_dir = os.path.join(temp_dir, "Structured3D")
            if os.path.exists(inner_dir):
                for item in os.listdir(inner_dir):
                    src = os.path.join(inner_dir, item)
                    dst = os.path.join(extract_dir, item)
                    shutil.move(src, dst)
                
                # Clean up
                shutil.rmtree(temp_dir)
                logger.info(f"Successfully extracted {zip_path}")
                return True
            else:
                logger.error("Expected inner directory 'Structured3D' not found")
                return False
                
    except zipfile.BadZipFile:
        logger.error(f"Error: {zip_path} is a corrupt ZIP file")
        return False
    except Exception as e:
        logger.error(f"An error occurred during extraction: {e}")
        return False

def download_and_extract(url: str, save_dir: str, extract_dir: str) -> bool:
    """Download and extract a single file.
    
    Args:
        url: URL to download from
        save_dir: Directory to save downloaded files
        extract_dir: Directory to extract files to
        
    Returns:
        bool: True if both download and extraction were successful
    """
    file_name = os.path.join(save_dir, os.path.basename(url))
    logger.info(f"Processing {url}")
    
    # Download the file
    if not download_file(url, file_name):
        return False
    
    # Extract the file
    if not extract_zip(file_name, extract_dir):
        return False
    
    # Clean up the downloaded zip file
    try:
        os.remove(file_name)
        logger.info(f"Cleaned up {file_name}")
    except Exception as e:
        logger.warning(f"Failed to clean up {file_name}: {e}")
    
    return True

def main():
    """Main function to download and extract all files."""
    # Setup directories
    save_dir = os.path.join("Data", "raw_S3D_perspective")
    extract_dir = os.path.join(save_dir, "structured3d_perspective")
    create_directories(save_dir, extract_dir)
    
    # Process each URL
    successful_downloads = 0
    for url in DOWNLOAD_URLS:
        if download_and_extract(url, save_dir, extract_dir):
            successful_downloads += 1
    
    # Report results
    total_files = len(DOWNLOAD_URLS)
    logger.info(f"Download and extraction complete! {successful_downloads}/{total_files} files processed successfully")

if __name__ == "__main__":
    main()
