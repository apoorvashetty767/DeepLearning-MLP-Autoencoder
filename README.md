# Deep Learning Project – MLP and Sparse Autoencoder

## 📌 Overview
This project implements a Multi-Layer Perceptron (MLP) and a Sparse Autoencoder from scratch using Python and NumPy on the MNIST dataset.

The project includes:
- Data loading and preprocessing
- MLP implementation and training
- Hyperparameter tuning
- Training loss and accuracy visualization
- Sparse autoencoder reconstruction
- Outlier detection using reconstruction error

---

## 📂 Project Files

### 🔹 Data Processing
- step1_load_mnist.py – Loads MNIST dataset  
- step2_flatten_mnist.py – Preprocesses and reshapes data  

### 🔹 MLP Implementation
- step3_mlp_numpy.py – Defines MLP architecture (784 → 128 → 10)  
- step4_mlp_train.py – Trains the MLP model  
- step4_mlp_tuning.py – Performs hyperparameter tuning  

### 🔹 Autoencoder Implementation
- step5_autoencoder.py – Sparse Autoencoder training (784 → 128 → 64 → 128 → 784)  
- step6_autoencoder_outlier.py – Outlier detection using reconstruction error  

---

## 📊 Outputs Generated

- Training loss curve  
- Training accuracy curve  
- Hyperparameter comparison table  
- Autoencoder reconstructed images  
- Outlier detection images  

---

## ▶ How to Run

Open terminal inside the project folder and run:

python step4_mlp_train.py

python step4_mlp_tuning.py

python step5_autoencoder.py

python step6_autoencoder_outlier.py


---

## 🛠 Requirements

- Python 3.x  
- NumPy  
- Matplotlib  

Install dependencies using:

pip install numpy matplotlib

---

## 📁 Repository Structure

DeepLearning-MLP-Autoencoder/
│
├── step1_load_mnist.py
├── step2_flatten_mnist.py
├── step3_mlp_numpy.py
├── step4_mlp_train.py
├── step4_mlp_tuning.py
├── step5_autoencoder.py
├── step6_autoencoder_outlier.py
├── README.md
└── Report.pdf
