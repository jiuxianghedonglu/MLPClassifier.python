# An implemention of MLP (Multi Layer Perceptron) Classifier from scratch in numpy.


# Description
- hidden layers activation is sigmoid
- output layer activation is softmax
- loss function is cross-entropy
- support save and load model weights
- setting hidden layers by paramater "hidden_sizes", examples: hidden_sizes=(100,),there is a hidden layer of 100 neurons;
hidden_sizes=(100,50),there are two hidden layers, 100 and 200 neurons.

# Example:

## Mnist Dataset Classfication

1. Download Data:

- [mnist_train.csv](https://pjreddie.com/media/files/mnist_train.csv)
- [mnist_test.csv](https://pjreddie.com/media/files/mnist_test.csv)

2. Run [example](https://github.com/jiuxianghedonglu/MLP-Classifier/blob/master/example.py)

3. [Training log of example](https://github.com/jiuxianghedonglu/MLP-Classifier/blob/master/mnist_example.png)