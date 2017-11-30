import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

classes = range(10)

data = np.load('data.npy')
X = data[()]['X']
y = data[()]['y']

# Visualize some examples from the dataset.
samples_per_class = 5
imgs = np.reshape(X, [-1, 16, 16])
labels = y

for j, cls in enumerate(classes):
    idxs = np.flatnonzero(np.equal(labels, cls))
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = j * samples_per_class + i + 1
        plt.subplot(samples_per_class, len(classes), plt_idx)
        plt.imshow(imgs[idx])
        plt.axis('off')
        plt.title(cls)
plt.show()

X_train = y_train = X_test = y_test = None


#######################################################################
# TODO:                                                               #
# Arrange the data into train and test sets                           #
# Be careful about how you split the data:                            #
# - If train and test distribution are very different your test       #
#   performance will be poor                                          #
# - Think about the sizes of the splits: What are good values and how #
#   does this affect your train/test performance?                     #
#######################################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.36)

#######################################################################
#                         END OF YOUR CODE                            #
#######################################################################


def pre_process(x):
    #######################################################################
    # TODO:                                                               #
    # Implement preprocessing of the data before feeding to the SVM.      #
    # NOTE: This function will be used to grade the performance on the    #
    # held-out test set                                                   #
    #######################################################################

   # x = sklearn.preprocessing.scale(x)
    x = sklearn.preprocessing.normalize(x)

    #scaler = StandardScaler().fit(x)
    #scaler.transform(x)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return x


X_train = pre_process(X_train)
X_test = pre_process(X_test)

print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)

from sklearn.svm import LinearSVC

def train_linear_SVM(X, y, C, max_iter=100):
    """
    Linear multi-class SVM solver.

    Args:
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train]
        C: Hyper-parameter for SVM
        max_iter: Maximum number of iterations

    Returns:
        lin_clf: The learnt classifier (LinearSVC instance)

    """
    lin_clf = None
    print('Solving linear-SVM...')

    #######################################################################
    # TODO:                                                               #
    #######################################################################

    lin_clf = LinearSVC(C=C, max_iter=max_iter)

    lin_clf.fit(X,y)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return lin_clf

C = 11.11
lin_clf = train_linear_SVM(X_train, y_train, C)

#######################################################################
# TODO:                                                               #
# Visualize the learnt weights (lin_clf.coef_) for all the classes:   #
# - Make a plot with ten figures showing the respective weights for   #
#   each of the classes                                               #
#######################################################################

for j, cls in enumerate(classes):
    idxs = np.flatnonzero(np.equal(labels, cls))
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = j * samples_per_class + i + 1
        plt.subplot(samples_per_class, len(classes), plt_idx)
        plt.imshow(np.reshape(lin_clf.coef_[j], [16,16]))
        plt.axis('off')
        plt.title(cls)
plt.show()



#######################################################################
#                         END OF YOUR CODE                            #
#######################################################################

from sklearn.metrics import accuracy_score, confusion_matrix

def eval_clf(y_pred_train, y_pred_test):
    acc_test = acc_train = cm_test = None
    #######################################################################
    # TODO:                                                               #
    # Use the learnt classifier to make predictions on the test set.      #
    # Compute the accuracy on train and test sets.                        #
    # Compute the confusion matrix on the test set.                       #
    #######################################################################

    acc_test = accuracy_score(y_test, y_pred_test)
    acc_train = accuracy_score(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)


    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return acc_train, acc_test, cm_test

acc_train, acc_test, cm_test = eval_clf(lin_clf.predict(X_train), lin_clf.predict(X_test))
print("Linear SVM accuracy train: {}".format(acc_train))
print("Linear SVM accuracy test: {}".format(acc_test))
print("Confusion matrix:\n%s" % cm_test)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def train_gaussian_SVM(X, y, C, gamma, max_iter=100):
    """
    Multi-class SVM solver with Gaussian kernel.

    Args:
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train]
        C: Hyper-parameter for SVM
        max_iter: Maximum number of iterations

    Returns:
        w: The value of the parameters after logistic regression

    """
    clf_rbf = None
    print('Solving RBF-SVM: This can take a while...')


    clf_rbf = SVC(C, 'rbf', gamma=gamma, max_iter=max_iter)

    clf_rbf.fit(X,y)

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return clf_rbf


C = 111.11 #112
gamma = 010.000000000000001 #10
clf_rbf = train_gaussian_SVM(X_train, y_train, C, gamma)

acc_train, acc_test, cm_test = eval_clf(clf_rbf.predict(X_train), clf_rbf.predict(X_test))
print("RBF SVM accuracy train: {}".format(acc_train))
print("RBF SVM accuracy test: {}".format(acc_test))
print("Confusion matrix:\n%s" % cm_test)


