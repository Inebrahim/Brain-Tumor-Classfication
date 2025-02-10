import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Concatenate, Input
import matplotlib.pyplot as plt
import time
import psutil
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# Dataset paths
dataset_path = r"C:\Users\Hp Aero\Documents\MS-AI\3rd semester\DL\project\archive (1)"
train_dir = os.path.join(dataset_path, "training")
test_dir = os.path.join(dataset_path, "testing")
print(f"Train directory: {train_dir}")

# Parameters
IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Data preparation
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_dir, target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=BATCH_SIZE, class_mode="categorical"
)

# Define a shared input layer
input_layer = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Load pretrained MobileNetV2
base_model1 = MobileNetV2(weights="imagenet", include_top=False, input_tensor=input_layer)
base_model1.trainable = False  # Freeze layers

# Load pretrained EfficientNetB0
base_model2 = EfficientNetB0(weights="imagenet", include_top=False, input_tensor=input_layer)
base_model2.trainable = False  # Freeze layers

# Create individual outputs
mobile_output = GlobalAveragePooling2D()(base_model1.output)
efficient_output = GlobalAveragePooling2D()(base_model2.output)

# Concatenate outputs
merged_output = Concatenate()([mobile_output, efficient_output])

# Add dense layers after merging
final_output = Dense(128, activation="relu")(merged_output)
final_output = Dropout(0.5)(final_output)
final_output = Dense(4, activation="softmax")(final_output)  # Adjust number of classes

# Define the blended model
model = Model(inputs=input_layer, outputs=final_output)

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

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,  # Use integer division
    validation_steps=test_generator.samples // BATCH_SIZE  # Use integer division
)


end_time = time.time()
training_time = end_time - start_time

cpu_usage_after = psutil.cpu_percent()
memory_usage_after = psutil.virtual_memory().used / (1024**3)

print(f"\nTraining Time: {training_time:.2f} seconds")
print(f"CPU Usage Before: {cpu_usage_before}% | After: {cpu_usage_after}%")
print(f"Memory Usage Before: {memory_usage_before:.2f} GB | After: {memory_usage_after:.2f} GB")

# Save the model
model.save("brain_tumor_cnn_simple.h5")
print("\nModel saved as 'brain_tumor_cnn_simple.h5'")

# Evaluate the model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Compute predictions
y_true = test_generator.classes
y_pred = np.argmax(model.predict(test_generator), axis=1)

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

# Compute classification metrics
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=list(test_generator.class_indices.keys())))

# Compute and display confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_generator.class_indices.keys()))
disp.plot(cmap="Blues")
plt.title("Confusion Matrix")
plt.show()
