import os
import time
import tensorflow as tf
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import psutil

# Correct dataset paths
train_dir = r"C:\Users\Hp Aero\Documents\MS-AI\3rd semester\DL\project\archive\Training (1)"
test_dir = r"C:\Users\Hp Aero\Documents\MS-AI\3rd semester\DL\project\archive\Testing (1)"

print(f"Train directory: {train_dir}")
print(f"Test directory: {test_dir}")

# Parameters
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 10 
LEARNING_RATE = 0.0001  # Lower learning rate for transfer learning

# Data Preparation
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False
)

# Load VGG19 Model
base_model = VGG19(include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze the convolutional base

# Build Model
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation="relu"),  # Fully connected layer
    Dropout(0.5),
    Dense(train_generator.num_classes, activation="softmax")  # Output layer
])

# Show model summary
print("\nModel Summary:")
model.summary()

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Measure training time and resources
start_time = time.time()
cpu_usage_before = psutil.cpu_percent()
memory_usage_before = psutil.virtual_memory().used / (1024**3)

# Train the model
history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    validation_steps=test_generator.samples // BATCH_SIZE
)

end_time = time.time()
training_time = end_time - start_time

cpu_usage_after = psutil.cpu_percent()
memory_usage_after = psutil.virtual_memory().used / (1024**3)

print(f"\nTraining Time: {training_time:.2f} seconds")
print(f"CPU Usage Before: {cpu_usage_before}% | After: {cpu_usage_after}%")
print(f"Memory Usage Before: {memory_usage_before:.2f} GB | After: {memory_usage_after:.2f} GB")

# Save the model
model.save("brain_tumor_vgg19.h5")
print("\nModel saved as 'brain_tumor_vgg19.h5'")

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy * 100:.2f}%")

# Plot training history
plt.figure(figsize=(10, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Get predictions and true labels
y_true = test_generator.classes  # True labels
y_pred_probs = model.predict(test_generator)  # Predicted probabilities
y_pred = np.argmax(y_pred_probs, axis=1)  # Predicted classes

# Compute classification metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=test_generator.class_indices.keys()))

# Compute and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_generator.class_indices.keys())
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
