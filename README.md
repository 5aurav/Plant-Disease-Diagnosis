# Plant-Disease-Diagnosis
A deep learning application that identifies diseases in potato plant leaves, classifying them as Early Blight, Late Blight, or Healthy.

<img src="images/Screenshot 2025-05-14 105412.png" alt="Project Structure" width="WIDTH" height="HEIGHT">

## Project Overview
This application uses a convolutional neural network (CNN) to classify potato leaf images into three categories:

+ Early Blight
+ Late Blight
+ Healthy

The model was trained on a dataset of potato leaf images and deployed as a web application using Flask.

## Dataset
Dataset - **PlantVillage** from Kaggle

+ Early Blight: 982
+ Late Blight: 1000
+ Healthy: 1064

## Technology Stack

+ **Backend**: Python, Flask
+ **Machine Learning**: TensorFlow, Keras
+ **Frontend**: HTML, CSS, JavaScript
+ **Data Processing**: NumPy, Matplotlib, Seaborn

## Features

+ Upload potato leaf images for instant disease detection
+ View confidence scores for predictions
+ View previous uploads and their classifications
+ Responsive web interface

##  Installation
### Prerequisites

+ Python 3.8+
+ TensorFlow 2.x

## Setup

Clone the repository
git clone https://github.com/yourusername/potato-disease-classifier.git
cd potato-disease-classifier

Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies
pip install -r requirements.txt

Download the model files

Ensure potato_disease_model_final_test.h5 and class_indices.json are in the project root directory

Run the application
python app.py

Open your browser and navigate to http://127.0.0.1:5000/

## Model Training
The model was trained using a CNN architecture with the following characteristics:

+ Four convolutional layers with batch normalization
+ Dropout layers to prevent overfitting
+ Data augmentation (rotation, zoom, flips)
+ Class weights to handle imbalanced data

To retrain the model:
python train.py
Note: You may need to modify the dataset paths in train.py to point to your training and validation data.
## Usage

+ Launch the application with python app.py
+ Click "Upload Image" to select a potato leaf image
+ View the prediction results including:
+     Disease classification
+     Confidence score
+     Probability distribution across all classes

## Project Structure
<img src="images/Screenshot 2025-05-14 210424.png" alt="Project Structure" width="WIDTH" height="HEIGHT">

## Model Performance
The model achieved:

+ Training accuracy: **~95%**
+ Validation accuracy: **~93%**
+ F1-score: **~0.92**

Detailed performance metrics can be found in the confusion matrix and classification report generated after training.
