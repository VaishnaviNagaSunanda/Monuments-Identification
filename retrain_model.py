import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout

# ---------- Dataset Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, "media", "Indian-monuments", "images", "train")
test_dir  = os.path.join(BASE_DIR, "media", "Indian-monuments", "images", "test")

print(f"Training data path: {train_dir}")
print(f"Testing data path: {test_dir}")

# ---------- Data Augmentation (improves accuracy) ----------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
)
test_datagen = ImageDataGenerator(rescale=1./255)

IMAGE_SIZE = (224, 224)  # MobileNetV2 works best at 224x224

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

num_classes = len(train_data.class_indices)
print(f"Number of monument classes: {num_classes}")
print(f"Classes: {list(train_data.class_indices.keys())}")

# ---------- MobileNetV2 Transfer Learning Model ----------
# Uses weights pre-trained on 1.2 million ImageNet images
base_model = MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,          # Remove the ImageNet classifier head
    weights='imagenet'          # Use pre-trained weights
)

# Freeze the base model layers (keep ImageNet features)
base_model.trainable = False

# Add our custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print(f"Model parameters: {model.count_params():,}")

# ---------- Phase 1: Train only the new head (5 epochs) ----------
print("\n=== Phase 1: Training classification head ===")
history1 = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data,
)

# ---------- Phase 2: Fine-tune the top layers of MobileNetV2 ----------
print("\n=== Phase 2: Fine-tuning top MobileNetV2 layers ===")
base_model.trainable = True

# Only fine-tune the last 30 layers
for layer in base_model.layers[:-30]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR for fine-tuning
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history2 = model.fit(
    train_data,
    epochs=5,
    validation_data=test_data,
)

final_acc = history2.history['val_accuracy'][-1]
print(f"\nFinal validation accuracy: {final_acc:.2%}")

# ---------- Save model as .h5 for cross-Keras compatibility ----------
os.makedirs("models", exist_ok=True)
model_save_path = os.path.join("models", "trained_model.h5")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")
