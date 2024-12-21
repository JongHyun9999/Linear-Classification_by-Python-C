# Neural Network Forward and Backward Propagation Code Repository

> PPT : https://docs.google.com/presentation/d/1xj0IgXs4lz6A-H1IErjPciLlPkjICw8Y/edit?usp=sharing&ouid=109216185127626877008&rtpof=true&sd=true

This repository contains foundational code written to understand the forward and backward propagation and equations of neural networks. 
It includes implementations in both C language and a library-free Python version. 
All implementations feature a simple 3-layer network, utilizing ReLU, Sigmoid, and Cross Entropy functions. 

The Python version is implemented to support mini-batch processing.

## Repository Structure

The repository is divided into the following directories:
1. `LogisticClassification`
2. `LogisticClassification_python`

- Directory 1 contains the version written in C.
- Directory 2 contains the version written in Python.

You can train the neural network by adjusting the hyperparameters declared as global variables. 
The training and test datasets are separated, and their specific compositions are as follows:
- Training Data: Seven lowercase alphabets (t, u, v, w, x, y, z) written in italics + Rotated images
- Test Data: Rotated images + Flipped images

The accuracy on the test dataset is poor when training without image augmentation. 
Therefore, augmented images are included in the training dataset. 
The decrease in accuracy for the test dataset is attributed to the model's exposure to untrained images (flipped images). 
Training with more diverse images is expected to improve the model's performance.
