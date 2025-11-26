# Deep-Learning-for-Image-Detection-Vehicle-vs-Non-Vehicle

## 1.0 INTRODUCTION
This project applies supervised deep learning to classify images as vehicle or non-vehicle using a Convolutional Neural Network (CNN). The model learns from a dataset of labeled images containing vehicles and non-vehicles. The system automates vehicle detection, supporting applications in autonomous driving, traffic monitoring, and security surveillance.

Key features of this project:

- Data preprocessing and augmentation (normalization, shear, zoom, horizontal flips)
- CNN classification (convolutional + pooling + fully connected layers)
- Model accuracy evaluation
- Prediction function for new images

## 2.0 DATASET
### LINK TO DATASET: https://drive.google.com/drive/folders/15nENLH9b2bIQQwiEy_QwunigYheUUIJ9
The dataset includes approximately 16,000 labeled images consisting of:
- Vehicle images
- Non-vehicle images
- Images cover diverse scenarios for better generalization
Dataset must be placed in the folder named: Vehicle-detection-data

## 3.0 RUNNING THE SCRIPT
Run the training script:
python src/train.py

The program will:
- Load the dataset
- Preprocess and augment images
- Train the CNN model
- Display training and validation accuracy & loss

## 4.0 PREDICTING NEW IMAGES
Run the testing script:
python src/test.py

The program will:
- Load an input image
- Preprocess the image
- Predict if the image contains a vehicle or not
- Display the predicted class
You may also pass your own image paths into the function for custom predictions.

## 5.0 REQUIREMENTS
tensorflow, numpy, scikit-learn, matplotlib, opencv-python

## 6.0 LICENSE
This project is for educational use.

## 7.0 AUTHORS & CREDITS
Iman A., Fitrah M.R., Zullaikha Z.
Created as part of Intelligent Systems group assignment.
