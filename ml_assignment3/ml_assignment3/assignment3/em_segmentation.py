import numpy as np
from sklearn.mixture import GaussianMixture

import time


def em_segmentation(img, k, max_iter=20):
    """
    Learns a MoG model using the EM-algorithm for image-segmentation.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of gaussians to be used

    Returns:
        label_img: A matrix of labels indicating the gaussian of size [h, w]

    """
    
    labels = None
    
    #######################################################################
    # TODO:                                                               #
    # 1st: Augment the pixel features with their 2D coordinates to get    #
    #      features of the form RGBXY (see np.meshgrid)                   #
    # 2nd: Fit the MoG to the resulting data using                        #
    #      sklearn.mixture.GaussianMixture                                #
    # 3rd: Predict the assignment of the pixels to the gaussian and       #  
    #      generate the label-image                                       #
    #######################################################################

    #1 preparations
    w = img.shape[1]
    h = img.shape[0]
    cols = img.shape[2]

    labels = img

    xgrid, ygrid = np.meshgrid(np.arange(0, w, 1), np.arange(0, h, 1))

    coordinates = np.stack((ygrid, xgrid), axis=2)
    img = np.concatenate((img, coordinates), axis=2)

    img = np.reshape(img, (h * w, cols + 2))

    #2 fit the MoG
    moG = GaussianMixture(n_components=k)
    moG.fit(img)


    img = moG.predict(img)

    labels = np.zeros([w * h, cols])
    means = np.delete(moG.means_, [3, 4], axis=1)

    for i in range(img.shape[0]):
        labels[i] = means[img[i]]

    labels = np.reshape(labels, (h, w, cols))

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return labels
                    