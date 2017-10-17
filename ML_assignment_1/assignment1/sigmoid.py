from numpy import exp, array, nditer


def sigmoid(z):
    """
    Computes the sigmoid of z element-wise.

    Args:
        z: An np.array of arbitrary shape

    Returns:
        g: An np.array of the same shape as z

    """

    g = None
    #######################################################################
    # TODO:                                                               #
    # Compute and return the sigmoid of z in g                            #
    #######################################################################

    g = array(z)

    for array_element in nditer(g, op_flags =['readwrite']):
        array_element[...] = sigmoid_for_single_value(array_element)


    pass

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return g

def sigmoid_for_single_value(value):
    numerator = 1;

    denumerator = 1 + exp(-value)

    return numerator / denumerator
