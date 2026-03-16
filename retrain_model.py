import os
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense

def plot_loss_curves(history, save_path):
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(len(loss))

    plt.figure(figsize=(10,4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.legend()
    plt.title('Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.legend()
    plt.title('Accuracy')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ---------- Dataset Paths ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
train_dir = os.path.join(BASE_DIR, "media", "Indian-monuments", "images", "train")
test_dir  = os.path.join(BASE_DIR, "media", "Indian-monuments", "images", "test")

print(f"Training data path: {train_dir}")
print(f"Testing data path: {test_dir}")

# ---------- Data Generators ----------
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=(300, 300),
    batch_size=32,
    class_mode='categorical'
)

# ---------- CNN Model ----------
model = Sequential([
    tf.keras.layers.Input(shape=(300, 300, 3)),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Conv2D(10, 3, activation='relu'),
    MaxPool2D(),
    Flatten(),
    Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# ---------- Train the Model ----------
history = model.fit(
    train_data,
    epochs=5,
    steps_per_epoch=len(train_data),
    validation_data=test_data,
    validation_steps=len(test_data)
)

# ---------- Save Model ----------
os.makedirs("models", exist_ok=True)
model_save_path = os.path.join("models", "trained_model.keras")
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# ---------- Save Training Curve ----------
plot_path = os.path.join("models", "training_plot.png")
plot_loss_curves(history, plot_path)
print(f"Plot saved to {plot_path}")
