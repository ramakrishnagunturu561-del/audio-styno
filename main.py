"""
AI-Based Secure Audio Steganography - Main Entry Point
======================================================
This script initializes the directory structure and launches the modern CustomTkinter GUI.
It guarantees that all necessary directories (outputs, models, dataset/normal, dataset/stego)
exist, and bootstraps a default training dataset using either the available original wave sample
or a generated synthetic signal.
"""

import os
from utils.logging_util import setup_logger
from utils import generate_synthetic_audio
from ml_detector import generate_stego_dataset

logger = setup_logger(__name__)

def initialize_directories():
    """
    Creates the outputs/, models/, and dataset/ directories if they do not exist.
    Populates the dataset with balanced training files if they are empty.
    """
    # Define directories
    dirs = [
        "outputs",
        "models",
        "dataset",
        "dataset/normal",
        "dataset/stego"
    ]
    
    # Create directories
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
    logger.info("Directory structure verified successfully.")
    
    # Check if dataset is empty
    normal_files = os.listdir("dataset/normal")
    stego_files = os.listdir("dataset/stego")
    
    if len(normal_files) == 0 or len(stego_files) == 0:
        logger.info("Dataset directories are empty. Bootstrapping training dataset...")
        
        # Check if the original sample is available in input/
        original_sample = "input/original_sample.wav"
        source_files = []
        
        if os.path.exists(original_sample):
            logger.info(f"Found default original sample at {original_sample}.")
            source_files.append(original_sample)
        else:
            # If not, generate a high-quality synthetic wave file to bootstrap
            bootstrap_file = "dataset/bootstrap_source.wav"
            logger.info(f"No original sample found. Generating synthetic bootstrap audio at {bootstrap_file}...")
            generate_synthetic_audio(bootstrap_file, duration_sec=5.0)
            source_files.append(bootstrap_file)
            
        try:
            # Generate 40 balanced segments (20 normal, 20 stego)
            generate_stego_dataset(source_files, dataset_dir="dataset", segments_count=40)
            logger.info("Successfully bootstrapped default training dataset.")
        except Exception as e:
            logger.error(f"Failed to bootstrap training dataset automatically: {e}")
        finally:
            # Clean up the bootstrap source file if generated
            if not os.path.exists(original_sample) and os.path.exists("dataset/bootstrap_source.wav"):
                os.remove("dataset/bootstrap_source.wav")
                logger.info("Cleaned up temporary bootstrap source wave.")

def main():
    """
    Initializes the workspace and launches the CustomTkinter GUI application.
    """
    print("=========================================================================")
    print("Starting AI-Based Secure Audio Steganography Suite...")
    print("=========================================================================")
    
    # Initialize workspace folders and dataset
    initialize_directories()
    
    # Import and run GUI
    try:
        from gui import AudioStegoApp
        app = AudioStegoApp()
        app.mainloop()
    except Exception as e:
        logger.critical(f"Unhandled exception during application launch: {e}", exc_info=True)
        print(f"\nCRITICAL ERROR: Failed to launch application: {e}")
        print("Please check that all dependencies are installed. Run: pip install -r requirements.txt")

if __name__ == "__main__":
    main()
