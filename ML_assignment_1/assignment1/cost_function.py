from sigmoid import sigmoid
import numpy as np


def cost_function(theta, X, y):
    """
    Computes the cost of using theta as the parameter for logistic regression

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        l: The cost for logistic regression

    """

    l = None
    #######################################################################
    # TODO:                                                               #
    # Compute and return the log-likelihood l of a particular choice of   #
    # theta.                                                              #
    #                                                                     #
    #######################################################################


    l = np.sum(np.dot(y, np.log(h_of_X_with_respect_to_theta(X,theta))) + np.dot(1-y, np.log(1- h_of_X_with_respect_to_theta(X, theta))))


    pass

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l

def h_of_X_with_respect_to_theta(X, theta): #works with arbitrary input as long as the dimensions are correct
    theta_dot_X = np.dot(X, theta)

    return sigmoid(theta_dot_X)


