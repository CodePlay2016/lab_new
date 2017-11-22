#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 09:12:21 2017

@author: codeplay2017
"""

from time import time
import pickle, os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC


def main(istrain=True):
    # Display progress logs on stdout
#    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    ###############################################################################
    # load the data on disk and load it as numpy arrays
    
    data_set_num = 7
    source_dir = ('/home/codeplay2017/code/lab'+
                  '/code/paper/realwork/python/resources/py2/data4trainset'+
                  str(data_set_num)+'/')
    train_num = 11
    test_num = 1
    t0 = time()
    trainset, testset = load_data(source_dir, 
                                  train_num, test_num, data_set_num)
    print("data loaded in %0.3fs" % (time() - t0))
    
    # introspect the images arrays to find the shapes (for plotting)
    n_samples, image_l = trainset.images.shape
    
    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X_train = trainset.images
    X_test = testset.images
    n_features = X_train.shape[1]
    
    # the label to predict is the id of the person
    y_train = np.where(trainset.labels==1)[1]
    y_test = np.where(testset.labels==1)[1]
    target_names = np.unique(y_train)
    n_classes = target_names.shape[0]
    
    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
  
    ##########################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 200
    
    print("Extracting the top %d PC from %d images"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    print('information ratio is '+str(pca.explained_variance_ratio_))
    print("pca established in %0.3fs" % (time() - t0))
    
#    eigenfaces = pca.components_.reshape((n_components, l))
    
    print("Projecting the input data on the PCs orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("pca done in %0.3fs" % (time() - t0))
    
    
    ###########################################################################
    # Train a SVM classification model
    
    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3],
                  'gamma': [0.005]}
    if istrain:
        clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
        clf = clf.fit(X_train_pca, y_train)
        with open('clf.pkl','wb') as f:
            pickle.dump(clf, f)
    else:    
        with open('clf.pkl', 'rb') as f:
            clf = pickle.load(f)
    print("training done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)
    
    
    ###########################################################################
    # Quantitative evaluation of the model quality on the test set
    
    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    print("prediction done in %0.3fs" % (time() - t0))
    print(type(y_pred))
    
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    return y_pred, y_test
    
#    prediction_titles = [title(y_pred, y_test, target_names, i)
#                     for i in range(y_pred.shape[0])]
#
#    plot_gallery(X_test, prediction_titles, h, w)
#    
#    # plot the gallery of the most significative eigenfaces
#    
#    eigenface_titles = ["eigenface %d" % i for i in range(eigenfaces.shape[0])]
#    plot_gallery(eigenfaces, eigenface_titles, h, w)
#    
#    plt.show()


###############################################################################
# prepare data
def load_data(source_dir, train_data_num, test_data_num, data_set_num):
    trainset = ImgDataSet()
    testset = ImgDataSet()
    for ii in range(train_data_num):
        file_name = os.path.join(source_dir, 'input_data'+str(data_set_num)+
                                 '_'+str(ii+1)+'.pkl')
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        trainset.join_data(data)
    for ii in range(test_data_num):
        file_name = os.path.join(source_dir, 'input_data'+str(data_set_num)+
                                 '_t_'+str(ii+1)+'.pkl')
        with open(file_name, 'rb') as f:
            data = pickle.load(f)
        testset.join_data(data)
    testset.make(shuffle=True,clean=True)
    trainset.make(shuffle=True,clean=True)
    return trainset, testset

def transform_label(label):
    return label.index(1)


###############################################################################
# Qualitative evaluation of the predictions using matplotlib

def plot_gallery(images, titles, h, w, n_row=3, n_col=4):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    for i in range(n_row * n_col):
        plt.subplot(n_row, n_col, i + 1)
        plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i], size=12)
        plt.xticks(())
        plt.yticks(())


# plot the result of the prediction on a portion of the test set

def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)

# this is the data structure for my dataset
class ImgDataSet():
    # arguments:
    # imagelist, labelist, the first dimension must be number of examples
    # these two lists must contain elements of numpy array type
    # if shuffle == True, all the data will be arranged randomly
    def __init__(self, image_array=np.array([]), label_array=np.array([]), shuffle=False):
        self.clean()
        self.images = np.array(self._imagelist)
        self.labels = np.array(self._labellist)
        if image_array.size:
            self.images = image_array[np.arange(image_array.shape[0])]
            self.labels = label_array[np.arange(label_array.shape[0])]
        if shuffle:
            self.shuffle()
        self._index_in_epoch = 0

    def add_data(self, img, label): 
        self._imagelist.append(list(img))
        self._labellist.append(list(label))
    
    # generally, when packaging data, when join the datasets, set clean=False
    # otherwise, set clean=True    
    def make(self, shuffle=True, clean=False):
        if len(self._imagelist):
            self.images = np.array(self._imagelist)
            self.labels = np.array(self._labellist)
        else:
            self.add_data(self.images, self.labels)
        if shuffle:
            self.shuffle()
        if clean:
            self.clean()
        else:
            self._imagelist = list(self.images)
            self._labellist = list(self.labels)

    def clean(self):
        self._imagelist = []
        self._labellist = []

    def shuffle(self):
        index = list(range(self.num_examples()))
        np.random.shuffle(index)
        self.images = self.images[index] # randomly arranged
        self.labels = self.labels[index] # randomly arranged

    def num_examples(self):
        return self.images.shape[0] # return the first dimension, num of samples
    
    def next_batch(self, batchsize, shuffle=False):
        if self._index_in_epoch + batchsize >= self.num_examples():
            self._index_in_epoch = 0
            shuffle = True
        start = self._index_in_epoch
        end = start + batchsize
        if shuffle:
            self.shuffle()
        self._index_in_epoch += batchsize
        return self.images[range(start, end)], self.labels[range(start, end)]
    
    def seperate_data(self, sep=0.5):
        num = self.num_examples()
        train_num = int(num * sep)
        _tempImages = self.images
        _tempLabels = self.labels
        self.train = ImgDataSet(_tempImages[:train_num], _tempLabels[:train_num])
        self.test = ImgDataSet(_tempImages[train_num:], _tempLabels[train_num:])
        del _tempImages
        del _tempLabels
        self.train.make(shuffle=True,clean=True)
        self.test.make(shuffle=True,clean=True)
    
    def join_data(self, other):
        if self.num_examples() != 0:
            self.images = np.concatenate((self.images, other.images), axis=0)
            self.labels = np.concatenate((self.labels, other.labels), axis=0)
        else:
            self.images = other.images[np.arange(other.num_examples())]
            self.labels = other.labels[np.arange(other.num_examples())]
    
    def isEmpty(self):
        return True if self.num_examples() == 0 else False

if __name__ == '__main__':
    yp,yt = main(istrain=True)



















