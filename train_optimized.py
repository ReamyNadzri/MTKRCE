import os
import shutil
import random
import logging
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import compute_class_weight

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Dataset Paths
SOURCE_DATA_DIR = 'images/'       # Folder containing original class subfolders
BASE_DATA_SPLIT_DIR = 'data_split/' # Folder where split data will be stored
TRAIN_DIR = os.path.join(BASE_DATA_SPLIT_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_SPLIT_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DATA_SPLIT_DIR, 'test')

# Split Ratios
TRAIN_RATIO = 0.84
VAL_RATIO = 0.08
TEST_RATIO = 0.08

# Model Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 20
MODEL_SAVE_PATH = 'kuih_recognition_model.keras'

def split_dataset(source, destination, train_r, val_r, test_r, force_resplit=False):
    """
    Splits images from source into train/val/test directories in destination.
    """
    if not os.path.exists(source):
        logger.error(f"Source directory '{source}' not found. Please ensure your 'images' folder exists.")
        return

    if os.path.exists(destination) and not force_resplit:
        logger.info(f"Split directory '{destination}' already exists. Skipping re-split.")
        return

    logger.info(f"Starting dataset split from '{source}' to '{destination}'...")

    # Ensure base directories exist
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)

    class_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})

    for class_name in os.listdir(source):
        class_source_path = os.path.join(source, class_name)
        if not os.path.isdir(class_source_path):
            continue

        # Create class subdirectories in split folders
        for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            os.makedirs(os.path.join(d, class_name), exist_ok=True)

        # Get and shuffle files
        files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
        if not files:
            continue
        
        random.shuffle(files)
        total = len(files)
        class_counts[class_name]['total'] = total

        # Calculate split indices
        train_end = int(total * train_r)
        val_end = train_end + int(total * val_r)

        # Assign files
        splits = {
            'train': (files[:train_end], TRAIN_DIR),
            'val': (files[train_end:val_end], VAL_DIR),
            'test': (files[val_end:], TEST_DIR)
        }

        # Copy files
        for split_name, (split_files, split_dir) in splits.items():
            for file_name in split_files:
                src = os.path.join(class_source_path, file_name)
                dst = os.path.join(split_dir, class_name, file_name)
                try:
                    shutil.copy2(src, dst)
                    class_counts[class_name][split_name] += 1
                except Exception as e:
                    logger.error(f"Error copying {file_name}: {e}")

    # Log results
    logger.info("\n--- Data Split Summary ---")
    for cls, counts in class_counts.items():
         logger.info(f"{cls:<15} | Total: {counts['total']:<4} | Train: {counts['train']:<4} | Val: {counts['val']:<4} | Test: {counts['test']:<4}")
    logger.info("--------------------------")

def run_training():
    # Execute the split
    split_dataset(SOURCE_DATA_DIR, BASE_DATA_SPLIT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, force_resplit=False)

    logger.info("Initializing Data Generators...")

    # Training generator with data augmentation
    datagen_train = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,      # Increased rotation
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,         # Increased zoom
        horizontal_flip=True,
        # vertical_flip=True,   # DISABLED: Usually mostly flat/top-down, but orientation matters for food "up"ness often.
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )

    # Validation/Test generator (rescaling only)
    val_test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    if not os.path.exists(TRAIN_DIR):
        logger.error("Train directory not found.")
        return

    # Flow from directories
    train_generator = datagen_train.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )

    validation_generator = val_test_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        TEST_DIR,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode='categorical',
        shuffle=False
    )

    num_classes = train_generator.num_classes
    class_labels = list(train_generator.class_indices.keys())
    logger.info(f"Classes detected ({num_classes}): {class_labels}")

    # --- Compute Class Weights ---
    logger.info("Computing class weights...")
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(train_generator.classes),
        y=train_generator.classes
    )
    class_weights = dict(enumerate(class_weights_array))
    logger.info(f"Class Weights: {class_weights}")

    # --- Model Building ---
    logger.info(f"Building model for {num_classes} classes using MobileNetV2 base...")
    base_model = MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    # Improved Architecture with BatchNormalization
    x = Dense(128, use_bias=False)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- Callbacks ---
    checkpoint = ModelCheckpoint('best_kuih_model_optimized.keras',
                                 monitor='val_accuracy',
                                 save_best_only=True,
                                 mode='max',
                                 verbose=1)

    early_stop = EarlyStopping(monitor='val_loss',
                               patience=5,
                               restore_best_weights=True,
                               verbose=1)

    reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                                  factor=0.2,
                                  patience=3,
                                  min_lr=1e-6,
                                  verbose=1)

    # --- Training ---
    logger.info("Starting training...")
    history = model.fit(
        train_generator,
        epochs=50,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights 
    )

    # --- Fine-Tuning ---
    logger.info("Starting Fine-Tuning...")
    base_model.trainable = True
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(1e-5),
                  metrics=['accuracy'])

    history_fine = model.fit(
        train_generator,
        epochs=20,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr],
        class_weight=class_weights
    )

    # --- Save Model ---
    model.save(MODEL_SAVE_PATH)
    logger.info(f"Model saved to {MODEL_SAVE_PATH}")

    # --- Plot Training History ---
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.tight_layout()
    plt.savefig('training_history.png')
    logger.info("Training history plot saved to 'training_history.png'")

    # --- Evaluation ---
    if test_generator and test_generator.samples > 0:
        logger.info("Evaluating on test set...")
        test_loss, test_acc = model.evaluate(test_generator)
        logger.info(f"Test Accuracy: {test_acc:.4f}")
        
        test_generator.reset()
        predictions = model.predict(test_generator)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = test_generator.classes
        
        report = classification_report(true_classes, predicted_classes, target_names=class_labels, zero_division=0)
        print(report)

        # Save Report to File
        with open('evaluation_results.txt', 'w') as f:
            f.write(f"Test Accuracy: {test_acc:.4f}\n")
            f.write(f"Test Loss: {test_loss:.4f}\n\n")
            f.write("--- Classification Report ---\n")
            f.write(report)
        logger.info("Evaluation results saved to 'evaluation_results.txt'")

        # Plot Confusion Matrix
        cm = confusion_matrix(true_classes, predicted_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label') # Added missing labels
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix_optimized.png')
        logger.info("Confusion matrix saved.")

if __name__ == "__main__":
    run_training()
