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


## Contact
If you have any comments, suggestions or anything you'd like to be clarify on, feel free
to reach me via [email](mailto:pmensah@aimsammi.org) or let's connect on [LinkedIn](https://www.linkedin.com/in/prince-mensah/).
