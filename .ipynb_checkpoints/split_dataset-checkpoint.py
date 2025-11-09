# split_dataset.py
import os
import shutil
import random
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# --- Configuration ---
SOURCE_DIR = 'images/'       # Directory with original class subfolders
BASE_OUTPUT_DIR = 'data_split/' # Base directory for train/val/test splits
TRAIN_RATIO = 0.84
VAL_RATIO = 0.08
TEST_RATIO = 0.08 # Ensure ratios sum to 1.0

if not abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9:
    raise ValueError("Train, validation, and test ratios must sum to 1.0")

def split_data():
    """
    Splits images from SOURCE_DIR into train, validation, and test sets
    in BASE_OUTPUT_DIR, maintaining class subdirectories.
    """
    if not os.path.exists(SOURCE_DIR):
        logging.error(f"Source directory '{SOURCE_DIR}' not found.")
        return

    # Create base output directories
    train_dir = os.path.join(BASE_OUTPUT_DIR, 'train')
    val_dir = os.path.join(BASE_OUTPUT_DIR, 'validation')
    test_dir = os.path.join(BASE_OUTPUT_DIR, 'test')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    logging.info(f"Output directories created/ensured at '{BASE_OUTPUT_DIR}'")

    class_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})

    # Iterate through each class directory in the source
    for class_name in os.listdir(SOURCE_DIR):
        class_source_path = os.path.join(SOURCE_DIR, class_name)

        if not os.path.isdir(class_source_path):
            logging.warning(f"Skipping non-directory item: {class_name}")
            continue

        # Create corresponding subdirectories in train, val, test
        class_train_path = os.path.join(train_dir, class_name)
        class_val_path = os.path.join(val_dir, class_name)
        class_test_path = os.path.join(test_dir, class_name)
        os.makedirs(class_train_path, exist_ok=True)
        os.makedirs(class_val_path, exist_ok=True)
        os.makedirs(class_test_path, exist_ok=True)

        # Get list of all image files for the current class
        try:
            files = [f for f in os.listdir(class_source_path)
                     if os.path.isfile(os.path.join(class_source_path, f))]
            
            if not files:
                 logging.warning(f"No files found in class directory: {class_source_path}")
                 continue # Skip empty directories

            random.shuffle(files) # Shuffle files randomly
            total_files = len(files)
            class_counts[class_name]['total'] = total_files

            # Calculate split indices
            train_end_idx = int(total_files * TRAIN_RATIO)
            val_end_idx = train_end_idx + int(total_files * VAL_RATIO)
            # Test gets the remainder

            # Assign files to splits
            train_files = files[:train_end_idx]
            val_files = files[train_end_idx:val_end_idx]
            test_files = files[val_end_idx:] # Remaining files go to test

            # Function to copy files
            def copy_files(file_list, destination_path, split_name):
                copied_count = 0
                for file_name in file_list:
                    source_file = os.path.join(class_source_path, file_name)
                    destination_file = os.path.join(destination_path, file_name)
                    try:
                        shutil.copy2(source_file, destination_file) # copy2 preserves metadata
                        copied_count += 1
                    except Exception as e:
                        logging.error(f"Failed to copy {source_file} to {destination_file}: {e}")
                class_counts[class_name][split_name] = copied_count

            # Copy files to respective directories
            logging.info(f"Processing class '{class_name}': Total={total_files}, Train={len(train_files)}, Val={len(val_files)}, Test={len(test_files)}")
            copy_files(train_files, class_train_path, 'train')
            copy_files(val_files, class_val_path, 'val')
            copy_files(test_files, class_test_path, 'test')

        except Exception as e:
            logging.error(f"Error processing directory {class_source_path}: {e}")


    logging.info("\n--- Split Summary ---")
    for class_name, counts in class_counts.items():
        logging.info(f"Class: {class_name} | Total: {counts['total']} | Train: {counts['train']} | Val: {counts['val']} | Test: {counts['test']}")
    logging.info("---------------------\nDataset splitting complete.")

if __name__ == '__main__':
    split_data()