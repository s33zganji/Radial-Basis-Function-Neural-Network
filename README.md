# Radial-Basis-Function-Neural-Network
Implementation of Radial Basis Function (RBF) neural network using evolution strategy (ES) algorithm with Python programming language, Computational Intelligence final course project, Spring 2019 <br/>
RBF neural network architecture is as follows: <br/>

We have L inputs with n dimentions for the first layer of the network which works as a radial basis function. In this layer, X which is the first feature space is converted to a new feature space, G, as follows: <br/>

Then after a linear transformation with weight matrix (w), new features are converted to a value in regression or classificaion mode.<br/>
The loss function and new weights can be calculated as below: <br/>

ES optimization algorithm generates chromosomes which minimize the error considering the loss function. <br/>

Therefore, we can use this trained neural network for classification and regression tasks. <br/>
Output of train and test for regression task: <br/>

Output of train and test for classification task: <br/>

