from sigmoid import sigmoid
import numpy as np


def gradient_function(theta, X, y):
    """
    Compute gradient for logistic regression w.r.t. to the parameters theta.

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        grad: The gradient of the log-likelihood w.r.t. theta

    """

    grad = None
    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of theta.              #
    # Compute the partial derivatives and set grad to the partial         #
    # derivatives of the cost w.r.t. each parameter in theta              #
    #                                                                     #
    #######################################################################

    h_X_theta = h_of_X_with_respect_to_theta(X, theta)
    grad = np.dot((y - h_X_theta), X)
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad

def h_of_X_with_respect_to_theta(X, theta): #works with arbitrary input as long as the dimensions are correct
    theta_dot_X = np.dot(X, theta)

    return sigmoid(theta_dot_X)