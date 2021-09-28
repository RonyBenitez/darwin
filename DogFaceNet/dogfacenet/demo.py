"""
DogFaceNet
The main DogFaceNet implementation
This file contains:
 - Data loading
 - Model definition
 - Model training

Licensed under the MIT License (see LICENSE for details)
Written by Guillaume Mougeot
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import os
import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
from online_training import *

#----------------------------------------------------------------------------
# Config.

PATH        = '../data/dogfacenet/aligned/after_4_bis/' # Path to the directory of the saved dataset
PATH_SAVE   = '../output/history/'                      # Path to the directory where the history will be stored
PATH_MODEL  = '../output/model/2019.07.29/'             # Path to the directory where the model will be stored
SIZE        = (224,224,3)                               # Size of the input images
TEST_SPLIT  = 0.1                                       # Train/test ratio

LOAD_NET    = True                                     # Load a network from a saved model? If True NET_NAME and START_EPOCH have to be precised
NET_NAME    = '2019.07.29.dogfacenet'                   # Network saved name
START_EPOCH = 135                                         # Start the training at a specified epoch
NBOF_EPOCHS = 250                                       # Number of epoch to train the network
HIGH_LEVEL  = True                                      # Use high level training ('fit' keras method)
STEPS_PER_EPOCH = 300                                   # Number of steps per epoch
VALIDATION_STEPS = 30                                   # Number of steps per validation
SAVE_TIME=15
#----------------------------------------------------------------------------
# Import the dataset.


print('Loading the dataset...')

filenames = np.empty(0)
labels = np.empty(0)
idx = 0
for root,dirs,files in os.walk(PATH):
    if len(files)>1:
        for i in range(len(files)):
            files[i] = root + '/' + files[i]
        filenames = np.append(filenames,files)
        labels = np.append(labels,np.ones(len(files))*idx)
        idx += 1
assert len(labels)!=0, '[Error] No data provided.'


alpha = 0.3
def triplet(y_true,y_pred):
    
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)

    return K.sum(tf.nn.relu(ap - an + alpha))

def triplet_acc(y_true,y_pred):
    a = y_pred[0::3]
    p = y_pred[1::3]
    n = y_pred[2::3]
    
    ap = K.sum(K.square(a-p),-1)
    an = K.sum(K.square(a-n),-1)
    
    return K.less(ap+alpha,an)

#----------------------------------------------------------------------------
# Model definition.
from scipy.spatial.distance import cdist

if LOAD_NET:
    print('Loading model from {:s}{:s}.{:d}.h5 ...'.format(PATH_MODEL,NET_NAME,START_EPOCH))

    model = tf.keras.models.load_model(
        '{:s}{:s}.{:d}.h5'.format(PATH_MODEL,NET_NAME,START_EPOCH),
        custom_objects={'triplet':triplet,'triplet_acc':triplet_acc})

    filenames_train=filenames[:20]
    labels_train=labels[:20]
    predict_train = model.predict_generator(predict_generator(filenames_train, 8),
                              steps=np.ceil(len(filenames_train)/8))
    matrix=cdist(predict_train,predict_train,"cosine")
    print(labels_train)
    print(np.round(matrix*100,0))
    
    
    print('Done.')
