"""
Model Setup Script

This script helps you set up the model directory with the required files
for the English-Urdu Neural Machine Translation app.

Usage:
  python setup_model.py --source /path/to/model/files

The script will create a 'model' directory in the current folder and
copy/rename the specified model files to the correct names.
"""

import os
import shutil
import argparse
from pathlib import Path

def setup_model_directory(source_dir):
    """
    Sets up the model directory with the required files.
    
    Args:
        source_dir: Path to directory containing model files
    """
    source_path = Path(source_dir)
    model_path = Path("model")
    
    # Create model directory if it doesn't exist
    model_path.mkdir(exist_ok=True)
    
    # Map of source files to destination files
    file_map = {
        "best_encoder.weights.h5": "best_encoder_weights.weights.h5",
        "best_encoder_weights.weights.h5": "best_encoder_weights.weights.h5",  # Already correct name
        "best_decoder.weights.h5": "best_decoder_weights.weights.h5",
        "best_decoder_weights.weights.h5": "best_decoder_weights.weights.h5",  # Already correct name
        "tokenizer_input.json": "tokenizer_input.json",
        "tokenizer_target.json": "tokenizer_target.json",
        "max_lengths.json": "max_lengths.json",
        "training_history.json": "training_history.json"
    }
    
    # Check if source directory exists
    if not source_path.exists() or not source_path.is_dir():
        print(f"Error: Source directory '{source_path}' does not exist or is not a directory")
        return False
    
    # Copy and rename files
    success = True
    for src_name, dst_name in file_map.items():
        src_file = source_path / src_name
        dst_file = model_path / dst_name
        
        # Check if source file exists (try variations of naming)
        if not src_file.exists():
            variations = [
                src_name,
                src_name.replace(".weights", ""),
                src_name.replace("weights.", ""),
                src_name.replace("best_", ""),
                f"best_{src_name}"
            ]
            
            found = False
            for var in variations:
                var_file = source_path / var
                if var_file.exists():
                    src_file = var_file
                    found = True
                    break
            
            if not found:
                print(f"Warning: Could not find '{src_name}' or variations in source directory")
                success = False
                continue
        
        # Copy file
        try:
            print(f"Copying {src_file} to {dst_file}")
            shutil.copy2(src_file, dst_file)
        except Exception as e:
            print(f"Error copying {src_file} to {dst_file}: {e}")
            success = False
    
    return success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up model directory for English-Urdu NMT app")
    parser.add_argument(
        "--source", "-s", 
        required=True, 
        help="Path to directory containing model files"
    )
    
    args = parser.parse_args()
    
    if setup_model_directory(args.source):
        print("\nModel directory setup complete!")
        print("You can now run the app with: streamlit run app.py")
    else:
        print("\nModel directory setup incomplete. Please check the warnings above.")