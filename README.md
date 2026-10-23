# CNN vs AlexNet-Inspired CNN: Image Classification

## Project Overview
This project compares a simple Convolutional Neural Network (CNN) with an AlexNet-inspired CNN for image classification using the CIFAR-10 dataset.

The models are evaluated based on test accuracy, validation accuracy, loss, and number of trainable parameters.

## Objectives
- Build a lightweight CNN architecture.
- Develop a deeper AlexNet-inspired CNN.
- Train both models using the same dataset.
- Compare model accuracy and complexity.
- Analyze the trade-off between performance and computational cost.

## Dataset
**CIFAR-10**

- 60,000 colour images
- Image size: 32 × 32 pixels
- 10 classes
- 50,000 training images and 10,000 test images

## Models

### 1. Simple CNN
- Three convolutional layers
- Max pooling
- Global average pooling
- Dense classification layer
- Dropout regularization

### 2. AlexNet-Inspired CNN
- Four convolutional layers
- Max pooling
- Flatten layer
- Two hidden dense layers
- Dropout regularization

*Note: This is a modified, AlexNet-inspired architecture adapted for CIFAR-10 images, not the original AlexNet.*

## Experimental Setup
- Optimizer: Adam
- Loss function: Sparse categorical cross-entropy
- Batch size: 64
- Training epochs: 5
- Validation split: 10%
- Random seed: 42

## Results

| Metric | Simple CNN | AlexNet-Inspired CNN |
|---|---:|---:|
| Test accuracy | 62.06% | 71.28% |
| Validation accuracy | 62.50% | 73.00% |
| Test loss | 1.0474 | 0.8389 |
| Parameters | 111,050 | 3,192,458 |

The AlexNet-inspired CNN achieved 9.22 percentage points higher test accuracy in this experiment. However, it contains approximately 28.75 times more parameters than the Simple CNN.

These results represent the initial five-epoch experiment. Further training and repeated experiments may produce different results.

## Project Structure

```text
cnn-alexnet-comparison/
├── dataset/
├── notebooks/
├── results/
│   ├── comparison.csv
│   └── accuracy_comparison.png
├── src/
│   ├── preprocessing.py
│   ├── simple_cnn.py
│   ├── alexnet_inspired.py
│   ├── train_compare.py
│   └── compare_results.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run

1. Install the required dependencies:

   `pip install -r requirements.txt`

2. Train and evaluate both models:

   `python src/train_compare.py`

3. Generate the accuracy comparison graph:

   `python src/compare_results.py`

## Conclusion
In the initial experiment, the AlexNet-inspired CNN achieved higher test accuracy than the Simple CNN, while requiring substantially more model parameters and computational effort. The experiment demonstrates the trade-off between model complexity and classification performance.

## Technologies Used
- Python
- TensorFlow / Keras
- Pandas
- Matplotlib
- CIFAR-10 dataset