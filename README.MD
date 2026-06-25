# Brain Tumor Classification using CNN

## 📌 Overview

This project implements a Convolutional Neural Network (CNN) for classifying brain MRI images into different tumor categories. The model is trained using TensorFlow and Keras and incorporates data augmentation, dropout regularization, and early stopping to improve generalization and reduce overfitting.

The objective of this project is to demonstrate the application of Deep Learning and Computer Vision techniques in medical image classification.

---

## 🚀 Features

* Brain MRI image classification
* Data augmentation for improved model robustness
* CNN-based architecture
* Early stopping to prevent overfitting
* Model saving and loading
* Single image prediction functionality
* Training and validation performance visualization

---

## 🛠️ Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Matplotlib

---

## 📂 Dataset Structure

```text
Training/
├── glioma
├── meningioma
├── pituitary
└── notumor

Testing/
├── glioma
├── meningioma
├── pituitary
└── notumor
```

---

## 🧠 Model Architecture

The CNN consists of:

* Conv2D (32 filters) + MaxPooling
* Conv2D (64 filters) + MaxPooling
* Conv2D (128 filters) + MaxPooling
* Flatten Layer
* Dropout (0.5)
* Dense Layer (128 neurons)
* Softmax Output Layer

---

## 📈 Training Configuration

| Parameter        | Value                    |
| ---------------- | ------------------------ |
| Image Size       | 150 × 150                |
| Batch Size       | 32                       |
| Epochs           | 20                       |
| Optimizer        | Adam                     |
| Loss Function    | Categorical Crossentropy |
| Validation Split | 20%                      |

---

## 📊 Results

* Training Accuracy: ~93%
* Validation Accuracy: ~77%
* Early Stopping implemented to improve generalization

The training curves indicate successful learning while also highlighting the challenges of overfitting in medical image classification tasks.

---

## 🔍 Sample Prediction

The model can predict the tumor class from a single MRI image and display:

* Predicted Class
* Confidence Score

Example:

```python
predict_image("sample_mri.jpg")
```

Output:

```text
Prediction: Glioma (92.4% confidence)
```

##

---

## 📁 Project Structure

```text
Brain-Tumor-Classifier/
│
├── main.py
├── requirements.txt
├── README.md
├── training_curves.png
├── brain_tumor_classifier_v3.keras
└── sample_predictions/
```

---

## 🔮 Future Improvements

* Transfer Learning using EfficientNet or ResNet
* Hyperparameter tuning
* Confusion Matrix and Classification Report
* Grad-CAM visualization for model interpretability
* Streamlit web application deployment

---

## 👩‍💻 Author

**Anshika Dwivedi**

B.Tech (Artificial Intelligence & Machine Learning)
