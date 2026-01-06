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
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# --- Configuration ---
# Dataset Paths
SOURCE_DATA_DIR = 'images/'
BASE_DATA_SPLIT_DIR = 'data_split/' # Using the same split directory as the notebook
TRAIN_DIR = os.path.join(BASE_DATA_SPLIT_DIR, 'train')
VAL_DIR = os.path.join(BASE_DATA_SPLIT_DIR, 'validation')
TEST_DIR = os.path.join(BASE_DATA_SPLIT_DIR, 'test')

# Split Ratios
TRAIN_RATIO = 0.84
VAL_RATIO = 0.08
TEST_RATIO = 0.08

# Model Parameters
IMG_SIZE = (299, 299) # IMPORTANT: InceptionV3 requires 299x299
BATCH_SIZE = 32
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'kuih_recognition_model_inceptionv3_enhanced.keras' # Changed name to avoid overwriting original immediately
EPOCHS_TO_TRAIN = 50 # Using 50 based on notebook logic (though variable was 20, fit used 50)

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
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        os.makedirs(d, exist_ok=True)
    class_counts = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0, 'total': 0})
    for class_name in os.listdir(source):
        class_source_path = os.path.join(source, class_name)
        if not os.path.isdir(class_source_path):
            continue
        for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
            os.makedirs(os.path.join(d, class_name), exist_ok=True)
        files = [f for f in os.listdir(class_source_path) if os.path.isfile(os.path.join(class_source_path, f))]
        if not files:
            continue
        random.shuffle(files)
        total = len(files)
        class_counts[class_name]['total'] = total
        train_end = int(total * train_r)
        val_end = train_end + int(total * val_r)
        splits = {
            'train': (files[:train_end], TRAIN_DIR),
            'val': (files[train_end:val_end], VAL_DIR),
            'test': (files[val_end:], TEST_DIR)
        }
        for split_name, (split_files, split_dir) in splits.items():
            for file_name in split_files:
                src = os.path.join(class_source_path, file_name)
                dst = os.path.join(split_dir, class_name, file_name)
                try:
                    shutil.copy2(src, dst)
                    class_counts[class_name][split_name] += 1
                except Exception as e:
                    logger.error(f"Error copying {file_name}: {e}")
    logger.info("\n--- Data Split Summary ---")
    for cls, counts in class_counts.items():
         logger.info(f"{cls:<15} | Total: {counts['total']:<4} | Train: {counts['train']:<4} | Val: {counts['val']:<4} | Test: {counts['test']:<4}")
    logger.info("--------------------------")

def create_model(num_classes):
    logger.info(f"Building model for {num_classes} classes using InceptionV3 base...")
    base_model = InceptionV3(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False
    logger.info("InceptionV3 base model loaded and layers frozen.")

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    logger.info("Custom top layers added.")
    return model

def main():
    # 1. Prepare Data
    split_dataset(SOURCE_DATA_DIR, BASE_DATA_SPLIT_DIR, TRAIN_RATIO, VAL_RATIO, TEST_RATIO, force_resplit=False)

    # 2. Generators
    logger.info("Initializing Data Generators...")
    # Training generator with data augmentation (Exact replica of notebook)
    datagen_train = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    # Validation/Test generator (only InceptionV3 preprocessing)
    val_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    if not os.path.exists(TRAIN_DIR):
        logger.error(f"Training directory not found: {TRAIN_DIR}. Aborting.")
        return

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
    logger.info(f"Detected {num_classes} classes: {class_labels}")

    # 3. Build Model
    model = create_model(num_classes)
    
    optimizer = Adam(learning_rate=LEARNING_RATE)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # model.summary() # Optional, can uncomment if needed

    # 4. Train
    checkpoint = ModelCheckpoint(
        MODEL_SAVE_PATH,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=10,
        mode='max',
        restore_best_weights=True,
        verbose=1
    )
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )

    logger.info("Starting training...")
    history = model.fit(
        train_generator,
        epochs=EPOCHS_TO_TRAIN,
        validation_data=validation_generator,
        callbacks=[checkpoint, early_stop, reduce_lr]
    )

    # 5. Save Training History Plot (New feature)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.tight_layout()
    plt.savefig('training_history_inceptionv3.png')
    logger.info("Saved training history plot to 'training_history_inceptionv3.png'")

    # 6. Evaluation
    logger.info("Evaluating on Test Set...")
    test_loss, test_acc = model.evaluate(test_generator)
    logger.info(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")

    predictions = model.predict(test_generator)
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes
    target_names = list(test_generator.class_indices.keys())

    # Classification Report
    report = classification_report(true_classes, predicted_classes, target_names=target_names)
    print("Classification Report:\n", report)

    # Save Evaluation Results (New feature)
    with open('evaluation_results_inceptionv3.txt', 'w') as f:
        f.write(f"Test Accuracy: {test_acc:.4f}\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write("--- Classification Report ---\n")
        f.write(report)
    logger.info("Saved evaluation results to 'evaluation_results_inceptionv3.txt'")

    # Confusion Matrix
    cm = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix_inceptionv3_enhanced.png')
    logger.info("Saved confusion matrix to 'confusion_matrix_inceptionv3_enhanced.png'")

if __name__ == "__main__":
    main()
