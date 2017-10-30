from sigmoid import sigmoid
import numpy as np


def predict_function(theta, X, y=None):
    """
    Compute predictions on X using the parameters theta. If y is provided
    computes and returns the accuracy of the classifier as well.

    """

    preds = None
    accuracy = None
    #######################################################################
    # TODO:                                                               #
    # Compute predictions on X using the parameters theta.                #
    # If y is provided compute the accuracy of the classifier as well.    #
    #                                                                     #
    #######################################################################
    
    preds = sigmoid(np.dot(X, theta))

    rounded_preds =np.round(preds)

    accuracy = 0.0

    for i in range(0, preds.shape[0]):
        if rounded_preds[i] == y[i]:
            accuracy += 1.0 / preds.shape[0]


    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return preds, accuracy