# data_utils.py (Updated to use separate folders)
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import logging

logger = logging.getLogger(__name__)

# --- Configuration ---
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
# ** Update Directory Paths **
BASE_DATA_DIR = 'data_split/' # Base directory created by split_dataset.py
TRAIN_DIR = os.path.join(BASE_DATA_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DATA_DIR, 'test')

def get_data_generators():
    """
    Creates data generators pointing to separate train, validation, and test directories.
    Applies augmentation only to the training set.
    """
    logger.info("Setting up data generators from split directories...")
    try:
        # Generator for Training (with augmentation)
        datagen_train = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Generator for Validation and Testing (only rescaling)
        datagen_val_test = ImageDataGenerator(rescale=1./255)

        # Check if directories exist
        if not os.path.exists(TRAIN_DIR):
             raise FileNotFoundError(f"Training directory '{TRAIN_DIR}' not found. Run split_dataset.py first.")
        if not os.path.exists(VAL_DIR):
             raise FileNotFoundError(f"Validation directory '{VAL_DIR}' not found. Run split_dataset.py first.")
        if not os.path.exists(TEST_DIR):
             raise FileNotFoundError(f"Test directory '{TEST_DIR}' not found. Run split_dataset.py first.")

        # Create Training Generator
        train_generator = datagen_train.flow_from_directory(
            TRAIN_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=True
        )

        # Create Validation Generator
        validation_generator = datagen_val_test.flow_from_directory(
            VAL_DIR,
            target_size=IMG_SIZE,
            batch_size=BATCH_SIZE,
            class_mode='categorical',
            shuffle=False # No need to shuffle validation data
        )

        # Create Test Generator
        test_generator = datagen_val_test.flow_from_directory(
            TEST_DIR,
            target_size=IMG_SIZE,
            batch_size=1, # Process one image at a time for evaluation
            class_mode='categorical',
            shuffle=False # CRUCIAL: Do not shuffle test data
        )

        if train_generator.samples == 0:
            logger.error("No training images found in the 'train' directory.")
            raise ValueError("No training images found.")
        if validation_generator.samples == 0:
            logger.warning("No validation images found in the 'validation' directory.")
        if test_generator.samples == 0:
            logger.warning("No test images found in the 'test' directory.")


        logger.info(f"Found {train_generator.samples} images for training in '{TRAIN_DIR}'.")
        logger.info(f"Found {validation_generator.samples} images for validation in '{VAL_DIR}'.")
        logger.info(f"Found {test_generator.samples} images for testing in '{TEST_DIR}'.")
        logger.info(f"Detected classes: {list(train_generator.class_indices.keys())}")

        return train_generator, validation_generator, test_generator

    except Exception as e:
        logger.error(f"Error creating data generators from split directories: {e}")
        return None, None, None

if __name__ == '__main__':
    # Example usage
    logging.basicConfig(level=logging.INFO)
    train_gen, val_gen, test_gen = get_data_generators()
    if train_gen:
        print("\nData generators from split directories created successfully.")
        print("Class Indices:", train_gen.class_indices)
    else:
        print("\nFailed to create data generators from split directories.")