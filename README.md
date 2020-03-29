# Introduction
Used convolutional neural network for classification of Dogs and Cats with an accuracy of 90% . 

# Datasets
Dataset was obtained from Kaggle.

link : https://www.kaggle.com/c/dogs-vs-cats/data

For the purpose of simplicity, a common folder named "dataset" is created which contains two directories named "Cat" and "Dog". The train and test data of both are combined and stored in their respective directory.

The complete dataset can be downloaded from the link and copied into the dataset folder with the above constraint. 


A folder named "random_test" contains some random images to check the efficiency of the model.

# How to Use
The **object_detection_preprocess** script is run to scan the images and create corresponding feature matrix and label vectors. Then the data is stored in pickle format. This reduces the time of scaning the training images repeatedly.

Now **object detection classifier** script is run to train and test the accuracy of our model. We can tune the hyperparameters to enhance the accuracy.

