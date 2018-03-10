# Cifar-Image-Classification
A 5 layer CNN for image classification on CIFAR 10 Dataset.

## Working
Baseline network structure can be summarized as follows:
1. Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation function.
2. Max Pool layer with size 2×2.
3. Dropout set to 10%.
4. Convolutional layer, 64 feature maps with a size of 3×3, a rectifier activation function 
5. Max Pool layer with size 2×2.
6. Dropout set to 20%
7. Convolutional layer, 128 feature maps with a size of 3×3, a rectifier activation function 
8. Max Pool layer with size 2×2.
9. Dropout set to 30%
10. Convolutional layer, 256 feature maps with a size of 3×3, a rectifier activation function 
11. Max Pool layer with size 2×2.
12. Dropout set to 40%
13. Convolutional layer, 512 feature maps with a size of 3×3, a rectifier activation function 
14. Max Pool layer with size 2×2.
15. Dropout set to 40%
16. Flatten layer.
17. Fully connected layer with 1024 units and a rectifier activation function.
18. Dropout set A logarithmic loss function is used with the stochastic gradient descent optimization algorithm configured with a large       momentum and weight decay start with a learning rate of 0.01.to 50%.
19. Fully connected output layer with 10 units and a softmax activation function.
20. A logarithmic loss function is used with the adam optimization algorithm. The model is trained for 20 epochs and batch size of 100. 

## Prerequisites
1. Python
2. Pandas
3. NLTK
4. Numpy
5. Sklearn 
6. Keras

## Built With
1. Jupyter Notebook
