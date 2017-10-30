from __future__ import print_function

import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from assignment1 import sigmoid, cost_function, gradient_function, logistic_Newton, logistic_SGD, gda, predict_function

# Set default parameters for plots
plt.rcParams['figure.figsize'] = (12.0, 6.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# Load the dataset
data = loadmat('../ML_assignment_1/faces.mat')
labels = np.squeeze(data['Labels'])
labels[labels == -1] = 0    # Want labels in {0, 1}
data = data['Data']

X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.3)
num_train = X_train.shape[0]
num_test = X_test.shape[0]

# As a sanity check, we print out the size of the training and test data.
print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

# Visualize some examples from the dataset.
samples_per_class = 10
classes = [0, 1]
train_imgs = np.reshape(X_train, [-1, 24, 24], order='F')


# for y, cls in enumerate(classes):
#     idxs = np.flatnonzero(np.equal(y_train, cls))
#     idxs = np.random.choice(idxs, samples_per_class, replace=False)
#     for i, idx in enumerate(idxs):
#         plt_idx = y * samples_per_class + i + 1
#         plt.subplot(len(classes), samples_per_class, plt_idx)
#         plt.imshow(train_imgs[idx])
#         plt.axis('off')
#         plt.title(cls)
# plt.show()

# Add intercept to X and normalize to range [0, 1]
X_train = np.concatenate((np.ones((num_train, 1)), X_train/255.), axis=1)
X_test = np.concatenate((np.ones((num_test, 1)), X_test/255.), axis=1)

# Test your sigmoid
# z_test = np.arange(-5, 5, 0.01)
# g_test = sigmoid(z_test)
# plt.plot(z_test, g_test)
# plt.title('Sigmoid')
# plt.show()

theta_0 = np.zeros(X_train.shape[1])
l_0 = cost_function(theta_0, X_train, y_train)
print('Log-likelihood with initial theta: ', l_0)

x_test = np.ones([2, 10])
theta_0 = np.zeros(10)
grad_0 = gradient_function(theta_0, x_test, 1.0)
print(grad_0)


method = 'sgd'

# We'll meausure the execution time
start = time.time()

if method is 'sgd':
    theta, losses = logistic_SGD(X_train, y_train)
elif method is 'newton':
    theta, losses = logistic_Newton(X_train, y_train)
elif method is 'gda':
    theta, losses = gda(X_train, y_train)
else:
    raise ValueError('Method not recognised!')

exec_time = time.time()-start
print('Total exection time: {}s'.format(exec_time))

if losses:
    plt.plot(losses)
    plt.title('Training Log-Likelihood')
    plt.show()

# We can have a look at what theta has learned to recognise as "face"
plt.imshow(np.reshape(theta[1:], [24, 24], order='F'))
plt.title('Learned theta')
plt.show()

# Test the final classifier
pred_test, accuracy_test = predict_function(theta, X_test, y_test)
pred_train, accuracy_train = predict_function(theta, X_train, y_train)
print('Test accuracy: {}'.format(accuracy_test))
print('Training accuracy: {}'.format(accuracy_train))

