from __future__ import print_function

#matplotlib inline

import collections
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from tensorflow.contrib.learn.python.learn.datasets import base

from time import time
import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC



from sklearn.datasets import fetch_lfw_people
lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.04, slice_=(slice(0, 250, None), slice(0, 250, None)))

#for name in lfw_people.target_names:
#	print(name)

# introspect the images arrays to find the shapes (for plotting)
n_samples, h, w = lfw_people.images.shape

# for machine learning we use the 2 data directly (as relative pixel
# positions info is ignored by this model)
X = lfw_people.data
n_features = X.shape[1]

# the label to predict is the id of the person
y = lfw_people.target
target_names = lfw_people.target_names
n_classes = target_names.shape[0]

print("Total dataset size:")
print("n_samples: %d" % n_samples)
print("n_features: %d" % n_features)
print("n_classes: %d" % n_classes)
print("h: %d" % h)
print("w: %d" % w)


# split into a training and testing set
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42)


images = X_train
cls = y_train

labels = np.zeros((cls.size, cls.max()+1))
labels[np.arange(cls.size),cls] = 1

Dataset = collections.namedtuple('Dataset', ['images', 'labels', 'cls'])
Datasets = collections.namedtuple('Datasets', ['train', 'test'])
#images = images.reshape(images.shape[0],images.shape[1], images.shape[2], 1)
train = Dataset(images=images, labels=labels, cls=cls)
#train = tf.contrib.learn.python.learn.datasets.mnist.DataSet(images, cls, one_hot=1, fake_data=1)

images = X_test
cls = y_test
labels = np.zeros((cls.size, cls.max()+1))
labels[np.arange(cls.size),cls] = 1
#images = images.reshape(images.shape[0],images.shape[1], images.shape[2], 1)
test = Dataset(images=images, labels=labels, cls=cls)
#test = tf.contrib.learn.python.learn.datasets.mnist.DataSet(images, cls, one_hot=1, fake_data=1)

data = Datasets(train=train, test=test)

#print(data)
#print(data.train)
#print(data.train.cls)


