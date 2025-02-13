# Brain Tumor Classification using Deep Learning

## Overview
This project focuses on classifying brain tumors using deep learning architectures, leveraging MRI-based tumor images. The goal is to create a model that can accurately classify whether an MRI scan contains a tumor and, if so, the type of tumor.

The project explores different deep learning models, including custom Convolutional Neural Networks (CNN), Transfer Learning using pre-trained models like VGG16, and a combination of EfficientNetB0 and MobileNetV2. The performance is evaluated in terms of accuracy, F1-scores, and training time.

## Key Features
- **Deep Learning Models**: 
  - Simple CNN
  - VGG16 (Transfer Learning)
  - EfficientNetB0 & MobileNetV2 (Blended Model)
- **Libraries Used**:
  - TensorFlow/Keras
  - PyTorch
  - OpenCV
  - Scikit-learn
- **Performance Metrics**:
  - Accuracy
  - F1-Score
  - Training Time
- **Dataset**: MRI scan images of brain tumors for classification.

## Requirements
To run this project, you need the following libraries:
- Python 3.x
- TensorFlow 2.x
- Keras
- PyTorch
- Scikit-learn
- OpenCV
- NumPy
- Matplotlib
- Pandas

You can install the dependencies using pip:
```bash
pip install tensorflow keras torch scikit-learn opencv-python numpy matplotlib pandas
```

## Dataset
The dataset used for training and testing the models consists of MRI images of brain tumors, which are typically categorized into classes such as:
- **No Tumor**
- **Glioma**
- **Meningioma**
- **Pituitary Tumor**

You can download the dataset from the [dataset link](add-dataset-link-here).

## Models and Results

### 1. **Simple CNN**
- **Accuracy**: 95.42%
- **F1-Score**: High across all classes
- **Training Time**: 1513 seconds (fastest)
- **Description**: A custom CNN model with multiple convolutional layers, followed by pooling layers and a fully connected layer at the end.

### 2. **VGG16 (Transfer Learning)**
- **Accuracy**: 82.61%
- **F1-Score**: Moderate across all classes
- **Training Time**: 4040 seconds (high computational cost)
- **Description**: A pre-trained VGG16 model fine-tuned on the brain tumor dataset.

### 3. **EfficientNetB0 & MobileNetV2 (Blended Model)**
- **Accuracy**: 90.77%
- **F1-Score**: Balanced performance across classes
- **Training Time**: Optimized with a combination of both architectures
- **Description**: A blended model combining **EfficientNetB0** and **MobileNetV2** for a more efficient architecture, which balances performance and resource utilization.

## Evaluation
- **Simple CNN** showed the best performance in terms of **accuracy** and **F1-scores**, while being the **most efficient in terms of training time**.
- **VGG16** offered a robust model with good generalization but came with a higher computational cost.
- **EfficientNetB0 & MobileNetV2** offered a balance between accuracy and efficiency, with **90.77% accuracy** and solid class-wise performance.

## How to Run
1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/brain-tumor-classification.git
   ```
2. Navigate to the project directory:
   ```bash
   cd brain-tumor-classification
   ```
3. Run the main script to start training:
   ```bash
   python train_model.py
   ```

   You can modify the script to choose which model (Simple CNN, VGG16, EfficientNetB0 & MobileNetV2) to train, or use pre-trained weights.

4. The model will save the trained weights in the `models/` directory. You can then use them to make predictions on new MRI images.

## Evaluation Metrics
- **Accuracy**: Measures the overall correctness of the model.
- **F1-Score**: Measures the balance between precision and recall, especially useful for imbalanced datasets.
- **Training Time**: Time taken to train the model for a given number of epochs.

## Acknowledgments
- The project uses [Keras](https://keras.io/), [TensorFlow](https://www.tensorflow.org/), [PyTorch](https://pytorch.org/), and other popular libraries for deep learning.
- Dataset credits to the creators for providing the MRI images of brain tumors.
