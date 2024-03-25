# Basic Machine Learning Algorithms
### About
This project is about the implementation of some of the basic machine learning
algorithms from scratch in Python.

## Table of Contents
- [Basic Machine Learning Algorithms](#machine-learning-from-scratch)
  * [About](#about)
  * [Table of Contents](#table-of-contents)
  * [Dependencies](#dependencies)
  * [Algorithms](#tests)
    + [Linear Regression](#linear-regression)
    + [Logistic Regression](#logistic-regression)
    + [Optimization](#optimization)
    + [Simple Neural Network](#simple-neural-network)
    + [Principal Component Analysis](#principal-component-analysis)
  * [Contact](#Contact)

## Dependencies

Having anaconda installed is preferable for running the program since it includes a 
large collection of pre-installed packages (not mandatory though). Aside that, no 
external installation is required beyond the standard Python libraries. However, this 
project uses numpy for numerical operations and Matplotlib for plotting, 
and scikit-learn for generating a synthetic dataset and splitting it. 
Ensure you have these libraries installed:

```bash
pip install numpy matplotlib scikit-learn
```

## Algorithms
### Linear Regression
This is a basic implementation of the linear regression algorithm from scratch in python. It is built to showcase
fundamental concepts including model fitting, prediction, evaluation and plots to demonstrate it's
performance after each iterative process. The implementation also includes, calculating the MSE and the coefficient 
of determination (R² score) for model evaluation.

    Follow theses steps to run the program:

    $ git clone https://github.com/pmensah28/Basic-ML-Algorithms.git
    $ cd Basic-ML-Algorithms/tests
    $ python3 linear_reg_test.py

<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/linear_reg_error.png" width="540"\>
</p>

<p align="center">
    Figure: Training progress of the linear regression model.
</p>
<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/linear_reg_pred.png" width="540"\>
</p>
<p align="center">
    Figure: Training progress of the linear regression model.
    <br> MSE: 104.20226976850218
    <br> Accuracy: 0.9538933013434353
</p>


### Logistic Regression
The Logistic regression algorithm was implemented with the sigmoid function, cross-entropy loss, and gradient descent for optimization.
The sigmoid function converts real-valued numbers between 0 and 1 which makes it suitable for binary classification. 
The cross-entropy loss measures the performance of the model whose output value is a probability values between 0 and 1. 
Finally, the gradient descent algorithm was used to minimize the cross-entropy loss function by updating the weights and bias.


    Follow theses steps to run the program:

    $ git clone https://github.com/pmensah28/Basic-ML-Algorithms.git
    $ cd Basic-ML-Algorithms/tests
    $ python3 logistic_reg_test.py

### Optimization
This method was implemented for both linear regression and logistic regression with different optimization techniques
including; the mini-batch gradient descent (MBGD), stochastic gradient descent (SGD) both with and without momentum.

Before testing this algorithm, be sure to select your preferred optimization technique with or without moment in
momentum in the code implementations.

    Follow theses steps to run the program:

    $ git clone https://github.com/pmensah28/Basic-ML-Algorithms.git
    $ cd Basic-ML-Algorithms/tests
    $ python3 optimization_test.py

### Simple Neural Network
The simple neural network is built from scratch with 1 input layer (2 neurons), 1 hidden layer (10 neurons), and 1 output layer (1 neuron).
Since it is a simple network, the sigmoid function was used in both the hidden layer and output layer. I computed the forward pass and applied the
cross-entropy loss function to compute the errors of the model since we are dealing with a classification  problem. To get the best parameter that
optimizes the performance of the model, I applied the back propagation technique (chain rule) in the learning process. Finally, I applied the gradient 
descent algorithm to update the parameters of the network.

    Follow theses steps to run the program:

    $ git clone https://github.com/pmensah28/Basic-ML-Algorithms.git
    $ cd Basic-ML-Algorithms/tests
    $ python3 simple_nn_test.py
<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/d_boundary1.png" width="540"\>
</p>
<p align="center">
    Figure: Plot showing the initial decision boundary of the nn model
</p>
<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/d_boundary2.png" width="540"\>
</p>
<p align="center">
    Figure: Plot showing the final decision boundary of the nn model
</p>
<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/nn_loss_plot.png" width="540"\>
</p>
<p align="center">
    Figure: Training and testing progress of the simple nn model.
    <br> Train accuracy: 1.0
    <br> Test accuracy: 1.0
</p>

### Principal Component Analysis
PCA is statistical technique that helps us to transform a dataset of higher 
dimensions into lower dimensions. This method utilizes orthogonal transformation to convert a set of observations
of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.
I fitted the PCA algorithm on the irish dataset to find the principal components. This transformation reduced the 
dimension of the data from 4 to a dimensionality of 2.

    Follow theses steps to run the program:

    $ git clone https://github.com/pmensah28/Basic-ML-Algorithms.git
    $ cd Basic-ML-Algorithms/tests
    $ python3 pca_test.py

<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/pca1.png" width="540"\>
</p>
<p align="center">
    Figure: Plot showing the number of components.
</p>
<p align="center">
    <img src="https://github.com/pmensah28/Basic-ML-Algorithms/blob/main/images/pca2.png" width="540"\>
</p>
<p align="center">
    Figure: Plot showing PCA reduced to two dimensions.
    <br> Transformed data shape: (150, 2)
    <br> Number of components: 2
    <br> Explained variance: [72.96 22.85  3.67  0.52]
    <br> Cumulative explained variance: [ 72.96  95.81  99.48 100.]
</p>

## Contact
If you have any comments, suggestions or anything you'd like to be clarify on, feel free
to reach me via [email](mailto:pmensah@aimsammi.org) or let's connect on [LinkedIn](https://www.linkedin.com/in/prince-mensah/).
