##Final Report
##Zamira Maulenova
##Group: BDA-2105

#Introduction
#Problem
The goal of this project is to create a model that can recognize human-drawn shapes from simple sketches. This work aims to train a machine learning model to classify images into categories.
Literature review
Image classification is the process of distributing and labeling groups of pixels or vectors in an image using certain algorithms. In this project, the method of supervised learning was used. This approach uses a pre-built training dataset of pictures that are already assigned to certain markers corresponding to categories (eg. birds, cars, hats, teeth, etc.). to create statistics that will be applied to the entire image. The model splits images into pixels and vectors to create statistics that will be applied to the entire image. The model splits images into pixels and vectors to create statistics that calculate the probability that each pixel belongs to a particular class. Finally, the pixels are labeled altogether with class features and show the probability that the entire image falls into one category or another. 
#Current work
In this project, an image classifier was developed. It was used to automatically label pictures drawn by hand on a special drawing box on a web page. A convolutional neural network was used to train a model based on a dataset of similar sketches. The Keras and TensorFlow libraries were used to create the model, and the backend of the program was based on TensorFlow.js (tfjs), which was obtained as a result of training the model in Google Colab.

#Data and Methods
#Information about the data
In this work, the Quick Draw dataset was used. This dataset contains 345 classes, in total containing about 50 million pictures of 28x28 pixels size. But due to memory limitations in Colab, only part of this dataset was used (100 classes and 5000 images in each class). The set of images for each class can be found on Google Cloud, where they are stored as numpy arrays. This model was trained on this dataset of labeled images, where each image is assigned a specific class.
 
 
#Description of the ML model
An image classification machine learning model is a type of algorithm that can automatically identify and categorize images based on their visual features. During the training process, the model learned to recognize patterns and features in the images that are associated with each class, and uses this knowledge to classify new, unlabeled images which are input by users.
There are several types of image classification models, including convolutional neural networks (CNNs), which are currently the most popular approach. CNNs use multiple layers of neurons to learn hierarchical representations of the image, where each layer detects increasingly complex features. The final layer of the network produces a set of probabilities for each possible class, and the class with the highest probability is chosen as the predicted label for the image.
The CNN model used in this project consists of 3 convolutional layers and 2 dense layers. It is made short and simple in order to maintain model’s ability to generalize data and avoid overfitting, as well as to make the model lightweighted and run fast in the browser, as it has to make predictions in real time.

#Results
The model achieved 91.96% accuracy on the test dataset and predicts the image class with high probability from real-life experience. To evaluate the accuracy, the built-in metric functions of the TensorFlow library were used.

#Discussion
To provide a critical review of the results of an image classification machine learning project, there are several key factors that should be considered:
Accuracy: The most important metric in an image classification project is the accuracy of the model. This refers to the percentage of images that the model correctly classified. This model shows very good accuracy performance, as it’s at 91.96%, while it is considered that ideally model should have over 90% accuracy rate.
Training and testing time: This model was trained within less than 1 minute, same time consumed for testing. It’s an important factor to consider, because a model that takes a long time to train may not be practical in a real-world setting.
Generalization: This model is able to correctly classify new images that it has not seen before with a high degree of accuracy.

#Resources
•	M Shinozuka, B Mansouri (2009) Synthetic aperture radar and remote sensing technologies for structural health monitoring of Civil Infrastructure Systems. Structural Health Monitoring of Civil Infrastructure Systems. https://www.sciencedirect.com/science/article/pii/B9781845693923500049 
•	Pralhad Gavali ME, J. Saira Banu (2019, August 2). Deep convolutional neural network for Image Classification on cuda platform. Deep Learning and Parallel Computing Environment for Bioengineering Systems. https://www.sciencedirect.com/science/article/pii/B9780128167182000130 

•	Training and evaluation with the built-in methods  :   Tensorflow Core. TensorFlow. (2022). https://www.tensorflow.org/guide/keras/train_and_evaluate 
•	Alyafeai, Z. (2018, September 27). Train a model in tf.keras with colab, and run it in the browser with tensorflow.js. Medium. https://medium.com/tensorflow/train-on-google-colab-and-run-on-the-browser-a-case-study-8a45f9b1474e 
