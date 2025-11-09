"""
model_trainer.py

ResNet50-based trainer and utilities expected by kuih_recognition.py.

Provides:
 - class KuihModelTrainer: high-level wrapper to train, evaluate and save the model.
 - function get_model_stats(): returns last-known metrics (accuracy/precision/recall/f1)
 - function check_and_retrain(): lightweight retrain trigger (placeholder)
 - function force_retrain(): immediate retrain helper

This module avoids importing kuih_recognition to prevent circular imports.
"""
import os
import logging
from datetime import datetime
import numpy as np
import tensorflow as tf

# Optional sklearn utilities
try:
    from sklearn.metrics import classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# Settings - keep in sync with main app if needed
MODEL_PATH = os.environ.get('KUIH_MODEL_PATH', 'kuih_recognition_model.keras')
IMAGES_PATH = os.environ.get('KUIH_IMAGES_PATH', 'images/')
TARGET_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-6
VALIDATION_TEST_SPLIT = 0.16  # 8% val, 8% test (split further in preprocessing)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# internal store for last run metrics
_model_stats = {
    'last_trained': None,
    'train_loss': None,
    'train_accuracy': None,
    'val_loss': None,
    'val_accuracy': None,
    'test_accuracy': None,
    'classification_report': None,
    'history': None
}


class KuihModelTrainer:
    def __init__(self,
                 images_path=IMAGES_PATH,
                 model_path=MODEL_PATH,
                 target_size=TARGET_SIZE,
                 batch_size=BATCH_SIZE,
                 epochs=EPOCHS,
                 lr=LEARNING_RATE):
        self.images_path = images_path
        self.model_path = model_path
        self.target_size = target_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.model = None
        self.class_names = None

    def build_model(self, num_classes):
        """
        Build a ResNet50 based model with a lightweight top.
        We freeze the base and train the head, then optionally fine-tune later.
        """
        logger.info("Building ResNet50-based model")
        base = tf.keras.applications.ResNet50(
            include_top=False,
            weights='imagenet',
            input_shape=self.target_size + (3,)
        )
        base.trainable = False  # freeze base initially

        inputs = tf.keras.Input(shape=self.target_size + (3,))
        x = tf.keras.applications.resnet.preprocess_input(inputs)
        x = base(x, training=False)
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dropout(0.4)(x)
        x = tf.keras.layers.Dense(256, activation='relu')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

        model = tf.keras.Model(inputs, outputs, name="kuih_resnet50")
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model
        logger.info("Model built successfully")
        return model

    def train(self, train_ds, val_ds, test_ds, class_names):
        """
        Train the model using provided datasets.

        Saves model to self.model_path and updates internal stats.
        """
        self.class_names = class_names
        num_classes = len(class_names)
        if self.model is None:
            self.build_model(num_classes)

        # Callbacks
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        ckpt_dir = os.path.join("checkpoints", now)
        os.makedirs(ckpt_dir, exist_ok=True)
        callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(ckpt_dir, "ckpt_{epoch:02d}.keras"),
                save_best_only=True,
                monitor='val_loss'
            ),
            tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
        ]

        logger.info("Starting training...")
        history = self.model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=self.epochs,
            callbacks=callbacks
        )

        # Save model
        try:
            self.model.save(self.model_path)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Failed to save model: {e}")

        # Evaluate on test_ds
        logger.info("Evaluating on test dataset...")
        test_loss, test_acc = self.model.evaluate(test_ds, verbose=1)
        logger.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

        # Optionally compute classification report if sklearn available
        class_report = None
        if SKLEARN_AVAILABLE:
            try:
                # Need predictions and true labels
                y_true = []
                y_pred = []
                for images, labels in test_ds.unbatch():
                    # labels is one-hot
                    y_true.append(int(tf.argmax(labels).numpy()))
                    img = tf.expand_dims(images, axis=0)
                    p = self.model.predict(img, verbose=0)
                    y_pred.append(int(np.argmax(p[0])))
                class_report = classification_report(y_true, y_pred,
                                                     target_names=class_names,
                                                     output_dict=True,
                                                     zero_division=0)
            except Exception as e:
                logger.warning(f"Could not compute classification report: {e}")

        # Update internal stats
        _model_stats.update({
            'last_trained': datetime.utcnow().isoformat(),
            'train_loss': float(history.history.get('loss')[-1]) if 'loss' in history.history else None,
            'train_accuracy': float(history.history.get('accuracy')[-1]) if 'accuracy' in history.history else None,
            'val_loss': float(history.history.get('val_loss')[-1]) if 'val_loss' in history.history else None,
            'val_accuracy': float(history.history.get('val_accuracy')[-1]) if 'val_accuracy' in history.history else None,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'classification_report': class_report,
            'history': history.history
        })

        return {
            'model_path': self.model_path,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'classification_report': class_report
        }

    def load_model(self):
        if os.path.exists(self.model_path):
            logger.info(f"Loading model from {self.model_path}")
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info("Model loaded")
                return self.model
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return None
        else:
            logger.warning(f"No model file found at {self.model_path}")
            return None

    def fine_tune(self, base_trainable_at=140, fine_epochs=5, train_ds=None, val_ds=None):
        """
        Unfreeze a portion of the ResNet50 base and fine-tune.
        base_trainable_at: layer index to start unfreezing
        """
        if self.model is None:
            self.load_model()
            if self.model is None:
                raise RuntimeError("Model must be loaded or trained before fine-tuning")

        # Find the base model inside the assembled model (ResNet50)
        for layer in self.model.layers:
            if isinstance(layer, tf.keras.Model) and 'resnet' in layer.name:
                base_model = layer
                break
        else:
            # fallback: assume the whole model's layers include base layers
            base_model = None

        if base_model is not None:
            logger.info("Unfreezing base model from layer %s", base_trainable_at)
            for layer in base_model.layers[:base_trainable_at]:
                layer.trainable = False
            for layer in base_model.layers[base_trainable_at:]:
                layer.trainable = True

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr/10)
            self.model.compile(optimizer=optimizer,
                               loss='categorical_crossentropy',
                               metrics=['accuracy'])

            history = self.model.fit(train_ds, validation_data=val_ds, epochs=fine_epochs)
            # Save fine-tuned model
            self.model.save(self.model_path)
            logger.info("Fine-tuned model saved")
            return history.history
        else:
            logger.warning("Could not locate base model for fine-tuning")
            return None


def get_model_stats():
    """Return last known model statistics."""
    return _model_stats.copy()


def check_and_retrain(feedback_count_threshold=50):
    """
    Placeholder function that would check for collected feedback and retrain
    if the threshold is reached. Returns bool indicating whether retrain started.

    In a real system this would query the feedback_log (database) or look at
    a feedback folder for new annotated images.
    """
    # This is intentionally a placeholder. The main application can implement
    # DB checks and call KuihModelTrainer.train() when needed.
    logger.info("check_and_retrain invoked - placeholder (no DB access here)")
    return False


def force_retrain(trainer: KuihModelTrainer, train_ds, val_ds, test_ds, class_names):
    """
    Immediately retrain using provided trainer and datasets.
    Returns training results dict from trainer.train()
    """
    logger.info("Force retrain invoked")
    return trainer.train(train_ds, val_ds, test_ds, class_names)


# If run directly, provide a simple local training entry point
if __name__ == "__main__":
    from preprocessing import build_datasets
    logger.info("Running standalone training demo (requires 'images/' folder)")
    train_ds, val_ds, test_ds, class_names = build_datasets(IMAGES_PATH, TARGET_SIZE, BATCH_SIZE)
    trainer = KuihModelTrainer(images_path=IMAGES_PATH, model_path=MODEL_PATH,
                               target_size=TARGET_SIZE, batch_size=BATCH_SIZE, epochs=EPOCHS)
    res = trainer.train(train_ds, val_ds, test_ds, class_names)
    logger.info("Training complete. Stats:")
    logger.info(res)