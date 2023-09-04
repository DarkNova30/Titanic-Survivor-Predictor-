# Titanic Survivor Predictor using Artificial Neural Networks (ANN) and Multilayer Perceptrons (MLP)

![Titanic](https://img.shields.io/badge/Titanic%20Survivor%20Predictor-ANN%20%26%20MLP-brightgreen.svg)
![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange.svg)

A machine learning project that predicts the likelihood of surviving the Titanic disaster using Artificial Neural Networks (ANN) and Multilayer Perceptrons (MLP). 

## Table of Contents
- [Introduction](#introduction)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Results](#results)


## Introduction
The Titanic Survivor Predictor is a machine learning project designed to predict whether a passenger aboard the Titanic survived or not based on various passenger attributes. It leverages the power of Artificial Neural Networks (ANN) and Multilayer Perceptrons (MLP) to analyze historical data and make predictions.

## Features
- Input features include passenger attributes like age, sex, ticket class, and more.
- Uses deep learning ANN and MLP models for survival prediction.
- Provides prediction probabilities and classification results.

## Model Architecture 
The ANN model is a type of deep learning architecture that consists of multiple layers of interconnected neurons. Here's an overview of the ANN architecture used in this project:

1. **Input Layer:** The input layer accepts the passenger attributes, including features like age, sex, ticket class, and more. Each feature is represented as a neuron in this layer.

2. **Hidden Layers:** The model includes two hidden layers, each with 32 and 64 number of neurons. These layers are responsible for feature extraction and pattern recognition. We use activation functions like ReLU (Rectified Linear Unit) to introduce non-linearity and enable the model to learn complex relationships within the data.
  
3. **Dropout Layer:** Each hidden layer is followed by a Dropout layer, denoted as "Dropout (rate=0.1)," where 0.1 represents the dropout rate. The Dropout layers randomly deactivate a fraction 0.1 of neurons during each training iteration. This helps prevent overfitting by reducing the model's reliance on specific neurons and encourages robust feature learning.

4. **Output Layer:** The output layer consists of a single neuron with a sigmoid activation function. It produces a prediction probability between 0 and 1, indicating the likelihood of a passenger surviving the Titanic disaster.

## Results
- Accuracy of ANN model: Testing accuracy=81.0%
