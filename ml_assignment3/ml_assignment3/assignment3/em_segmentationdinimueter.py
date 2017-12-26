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
        label_img: A matrix of label_img indicating the gaussian of size [h, w, 3]

    """

    label_img = None

    #######################################################################
    # TODO:                                                               #
    # 1st: Augment the pixel features with their 2D coordinates to get    #
    #      features of the form RGBXY (see np.meshgrid)                   #
    # 2nd: Fit the MoG to the resulting data using                        #
    #      sklearn.mixture.GaussianMixture                                #
    # 3rd: Predict the assignment of the pixels to the gaussian and       #
    #      generate the label-image                                       #
    #######################################################################

    x, y = np.meshgrid(np.arange(0, img.shape[1], 1), np.arange(0, img.shape[0], 1))
    image_matrix = np.concatenate((img, np.stack((y, x), axis=2)), axis=2)
    image_array = np.reshape(image_matrix, (img.shape[0] * img.shape[1], 5))

    MoG = GaussianMixture(k).fit(image_array)
    labels = MoG.predict(image_array)
    centers = np.delete(MoG.means_, [3, 4], axis=1)

    new_image_array = np.take(centers, labels, axis=0)
    label_img = np.reshape(new_image_array, (img.shape[0], img.shape[1], 3))

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return label_img
