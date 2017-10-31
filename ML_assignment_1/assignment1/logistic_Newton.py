from cost_function import cost_function
from gradient_function import gradient_function
from sigmoid import sigmoid
import numpy as np
import time


def logistic_Newton(X, y, num_iter=10):
    """
    Perform logistic regression with Newton's method.

    Args:
        theta_0: Initial value for parameters of shape [num_features]
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train, 1]
        num_iter: Number of iterations of Newton's method

    Returns:
        theta: The value of the parameters after logistic regression

    """

    theta = np.zeros(X.shape[1])
    losses = []
    for i in range(num_iter):
        start = time.time()
        #######################################################################
        # TODO:                                                               #
        # Perform one step of Newton's method:                                #
        #   - Compute the Hessian                                             #
        #   - Update theta using the gradient and the inverse of the hessian  #
        #                                                                     #
        # Hint: To solve for A^(-1)b consider using np.linalg.solve for speed #
        #######################################################################

        pass

        hess = hessian(theta, X)

        theta = theta -  np.linalg.solve(hess, gradient_function(theta, X, y))

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        exec_time = time.time()-start
        loss = cost_function(theta, X, y)
        losses.append(loss)
        print('Iter {}/{}: cost = {}  ({}s)'.format(i+1, num_iter, loss, exec_time))

    return theta, losses

def h_of_X_with_respect_to_theta(X, theta): #works with arbitrary input as long as the dimensions are correct
    theta_dot_X = np.dot(X, theta)

    return sigmoid(theta_dot_X)

def hessian(theta, X):

    h_of_X = h_of_X_with_respect_to_theta(X, theta)

    theta_diag = (np.multiply(h_of_X, 1 - h_of_X))

    x_T_dot_diag = theta_diag * (-X.T)

    return np.matmul(x_T_dot_diag, X)
