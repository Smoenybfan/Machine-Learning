from sklearn.cluster import KMeans
import numpy as np
import sklearn


def kmeans_colors(img, k, max_iter=100):
    """
    Performs k-means clusering on the pixel values of an image.
    Used for color-quantization/compression.

    Args:
        img: The input color image of shape [h, w, 3]
        k: The number of color clusters to be computed

    Returns:
        img_cl:  The color quantized image of shape [h, w, 3]

    """

    img_cl = None

    #######################################################################
    # TODO:                                                               #
    # Perfom k-means clustering of the pixel values of the image img.     #
    #######################################################################
    img = np.array(img, dtype=np.float64) / 255
    w,h,d = tuple(img.shape)
    image_array = np.reshape(img, (w * h, d))

    # Fitting model on a small sub-sample of the data
    image_array_sample = sklearn.utils.shuffle(image_array, random_state=0)[:1000]

    # In this case n_clusters is the amount of colors
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=max_iter).fit(image_array_sample)

    labels = kmeans.predict(image_array)

    # Reproduce the picture
    img_cl = np.zeros((w, h, d))
    label_idx = 0
    for i in range(w):
        for j in range(h):
            img_cl[i][j] = kmeans.cluster_centers_[labels[label_idx]]
            label_idx += 1
    
    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################

    return img_cl