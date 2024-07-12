
 # FAKE-CURRENCY-DETECTION-USING-ML(Logistic Regression)

Welcome to the **Fake Currency Detector using Logistic Regression** Repository!
This repository contains the code implementation of a fake currency detector using a logistic regression model. The project aims to identify and classify banknotes as either real or fake based on their features

![final img fake currency](https://github.com/user-attachments/assets/7812f8a7-77e9-408a-8c57-d5c38b11d46b)

## Overview

This project aims to test the authenticity of currency notes by preparing a system that uses machine-learning techniques to classify banknotes as real or fake based on their features. This currency authentication system has been designed completely using Python. The *scikit-learn* library has been used for the logistic regression model, and the *Tkinter* library has been used for building the GUI.

## Inspiration

Machine learning has revolutionized the way we approach problems and has opened up new possibilities for solving complex issues. One such problem is the detection of fake notes, which has been an ongoing challenge for financial institutions and businesses. It is possible to train models that can precisely predict the veracity of banknotes using machine learning. These algorithms can learn to recognize patterns and features that separate authentic notes from fakes by training them on vast datasets of both real and counterfeit notes. The way we approach challenges has been transformed by machine learning, which has also created new opportunities for resolving challenging problems. The detection of counterfeit notes is one such issue, which has been a persistent difficulty for businesses and financial institutions.

## How does it work?

The provided code implements a fake currency detector using a logistic regression model. It involves several steps, such as data loading, preprocessing, training, and evaluation of the model, and provides options to visualize the results. I'll explain each section of the code in detail:

## STEP 1:Libraries and Initialization

The code imports necessary libraries such as tkinter for GUI, pandas for data manipulation, scikit-learn for machine learning, and matplotlib and seaborn for visualization.
Initializes global variables for data, train-test splits, scaler, and the logistic regression model.

## STEP 2:GUI Setup

Creates the main window using tkinter and sets up the layout with labels, buttons, and a text area for displaying messages.

## STEP 3:Loading Dataset

Provides a button to upload a CSV file containing the dataset.
The load_dataset function reads the selected CSV file using pandas and displays the dataset length in the text area.

## STEP 4:Data Preprocessing

The preprocess_data function renames the dataset columns for ease of use and displays the first 10 rows of the dataset.

## STEP 5:Train-Test Split:

The train_test_split_data function splits the dataset into training and testing sets using scikit-learn's train_test_split function.
It also scales the features using StandardScaler.

## STEP 6:Training Logistic Regression Model:

The logistic_regression_accuracy function trains a logistic regression model using the scaled training data and computes its accuracy on the test data.
The accuracy is displayed in the text area.

## STEP 7:Visualization:

The fuel_consumption_graph and bar_graph functions generate bar graphs to visualize the distribution of the target variable (authenticity of the banknotes)seaborn and matplotlib are used for plotting.

## STEP 8:Prediction:

The print_logistic_regression_results function demonstrates how to use the trained model to predict the authenticity of a new banknote based on its features.
The result (original or fake) is displayed in the text area.

## Demo  
https://github.com/user-attachments/assets/ff0bae6d-888a-4f1e-8ccc-64602fa321a0

## Bar Graph
![Screenshot (58)](https://github.com/user-attachments/assets/29b9cb42-4dc4-4866-9cf2-799f4615d1fe)

## Data set
The dataset used in this project is downloaded from the UCI Machine Learning Repository's "Banknote Authentication Data Set". It consists of features extracted from images of banknotes, such as variance, skewness, curtosis, and entropy, and it is labeled to indicate whether a banknote is real or fake.

## Dependencies
      
The project requires the following libraries and dependencies:

- Python
- `tkinter` for GUI
- `pandas` for data manipulation
- `scikit-learn` for machine learning
- `seaborn` and `matplotlib` for visualization
  
## Installation

### Clone the repository:
```sh
git clone https://github.com/Keerthana1417/FAKE-CURRENCY-DETECTION.git


