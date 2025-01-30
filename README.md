# Image Captioning Model Using CNN-RNN

This repository contains the implementation of an image captioning model that utilizes a Convolutional Neural Network (CNN) and Recurrent Neural Network (RNN) architecture. The model is trained using the *Flickr8k* dataset, combining a *DenseNet* pretrained model for feature extraction and an *LSTM* for generating captions. The model is deployed using *Streamlit* for a simple web interface.
## Key Features
- *Dataset: The model is trained on the **Flickr8k dataset*, a collection of 8,000 images with 5 captions per image.
- *Feature Extraction: The model uses **DenseNet* as a pretrained CNN for extracting image features. Specifically, the second-to-last layer of DenseNet is used to capture the features.
- *Caption Generation: The model uses an **LSTM* (Long Short-Term Memory) network to generate captions based on the extracted features from the CNN.
- *Deployment: The model is deployed using **Streamlit* in PyCharm for an interactive web application.

## Installation

To use this model, make sure you have the following libraries installed:

bash
pip install tensorflow keras numpy matplotlib pillow streamlit


## Dataset
- The *Flickr8k dataset* can be downloaded from [this link](https://github.com/jbrownlee/Datasets/blob/master/Flickr8k_Dataset.txt).
- Place the images and corresponding captions in the appropriate directories as per the project structure.

## Model Architecture
- *DenseNet*: The model uses a pretrained DenseNet for feature extraction. Only the second-to-last layer is used to capture the relevant features from the images.
- *LSTM*: The extracted image features are fed into an LSTM model to generate captions in the form of text.

## Deployment with Streamlit

To run the Streamlit app locally, follow these steps:

1. Navigate to the project directory.
2. Run the following command in your terminal:

bash
streamlit run app.py


This will launch a local server and open the app in your default web browser, allowing you to interact with the model by uploading images and generating captions.

