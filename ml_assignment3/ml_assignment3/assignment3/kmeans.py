import numpy as np
import time


def kmeans(X, k, max_iter=100):
    """
    Perform k-means clusering on the data X with k number of clusters.

    Args:
        X: The data to be clustered of shape [n, num_features]
        k: The number of cluster centers to be used

    Returns:
        centers: A matrix of the computed cluster centers of shape [k, num_features]
        assign: A vector of cluster assignments for each example in X of shape [n] 

    """

    centers = None
    assign = None

    start = time.time()

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the input data X and store the         #
    # resulting cluster-centers as well as cluster assignments.           #
    #                                                                     #
    #######################################################################

    # 1st step: Chose k random rows of X as initial cluster centers

    centers = X[np.random.randint(0, X.shape[0], k)]
    assign = np.zeros(X.shape[0])

    for l in range(max_iter):
        prev_assign = np.array(assign)

        # 2nd step: Update the cluster assignment
        for i in range(X.shape[0]):
            x = X[i]
            diff =  np.tile(x, (k, 1)) - centers
            assign[i] = np.argmin(np.linalg.norm(diff, None, axis=1))

        # 3rd step: Check for convergence

        if np.array_equal(prev_assign, assign):
            break

        # 4th step: Update the cluster centers based on the new assignment

        for j in range(centers.shape[0]):

            numinator = np.where(assign == j, np.ones(assign.shape[0]), np.zeros(assign.shape[0]))

            denumiator = np.sum(numinator)

            numinator = np.dot(numinator,X)

            centers[j] = numinator / denumiator


    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    exec_time = time.time() - start
    print('Number of iterations: {}, Execution time: {}s'.format(i + 1, exec_time))

    return centers, assign
