# Skin Cancer Detection Using Deep Learning

## Overview
This project uses deep learning techniques to classify skin lesions as either **benign** or **malignant**. It includes:
- **Data Preprocessing**: Preparing image datasets for training.
- **Model Training**: Using a pre-trained InceptionV3 deep learning model.
- **Evaluation**: Testing the model's accuracy.
- **Graphical User Interface (GUI)**: A tool for uploading images and getting predictions.
## Project Structure
skin-cancer-detection/
├── preprocessing.py         # Script for data preprocessing
├── train_model.py           # Script to train the model
├── evaluate_model.py        # Script to test and evaluate the model
├── gui.py                   # GUI to make predictions using the trained model
├── README.md                # Documentation 
├── models/                  # Folder for saving the trained model
│   └── model.h5             # Trained model file
## Requirements
To run this project, you’ll need:
- **Python 3.8+**
- TensorFlow/Keras
- Pillow
- NumPy
- Pandas
- tkinter (for GUI)

Install dependencies using:
```bash
pip install -r requirements.txt
### 1. Data Preparation
- **Note**: The dataset is not included in this repository.
- To use this project, you need to prepare your own image dataset in the following folder structure:
data/ ├── train/ │ ├── nevus/ │ ├── melanoma/ │ ├── seborrheic_keratosis/ ├── validation/ ├── test/

- Place your image data into the respective folders and adjust the paths in the `preproces
### 2. Preprocess the Data
1. After organizing your dataset, run the `preprocessing.py` script to generate CSV files for your dataset:
   ```bash
   python preprocessing.py
---
### 3. Train the Model
1. Open `train_model.py` and ensure the dataset paths are correct.
2. Train the model using:
   ```bash
   python train_model.py
---

### 4. Evaluate the Model
1. Test the model on your dataset using:
   ```bash
   python evaluate_model.py

### 5. Run the GUI
1. Ensure `model.h5` (the trained model) exists in the `models/` folder.
2. Launch the GUI with:
   ```bash
   python gui.py
### GUI Prediction
- **Uploaded Image**: The uploaded image will be displayed in the GUI.
- **Prediction**: The result will be shown as "Malignant" or "Benign".

## Notes
- **Dataset Not Included**: This repository does not contain any image datasets. Users must prepare their own data.
- **Trained Model Not Included**: The `model.h5` file is not included. You can generate it by running `train_model.py` after preparing your dataset.

## Contact
For questions or suggestions:
- **Author**: Mobina Azimi
- **Email**: mobinaazimi999@gmail.com


