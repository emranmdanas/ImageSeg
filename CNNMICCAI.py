import os
import random
import tensorflow as tf
from tensorflow.contrib import layers
from tensorflow.contrib.layers.python.layers import initializers
from tensorflow.contrib import losses
from tensorflow.contrib.framework import arg_scope
import time
#import wget
import tarfile
import numpy as np
import cv2
import logging
from math import ceil
# from skimage import filter
from PIL import Image
from PIL import ImageFilter
import scipy.io
import matplotlib.pyplot as plt
#from skimage import filter
import scipy.io as io






Feature_layer1=64      # Number of feature maps in layer 1  
Feature_layer2=96     # Number of feature maps in layer 2
Feature_layer3=128     # Number of feature maps in layer 3
Feature_layer4=128     # Number of feature maps in layer 4
Feature_layer5=128     # Number of feature maps in layer 5
Feature_layer6=128      # Number of feature maps in layer 6
batch_size = 4  # Mini-batch size

weightfactor = 3 # weightfactor for the amplitude where you want to provide more weight. Recommend to use a small validation set to fix this hyper-parameter
num_classes=2          # Number of classes: 2 (prostate and background) 

inputdims=224           # Size of input and output images
initial_learning_rate=1e-4

def _resnet_layer(bottom, filters, scopename, dilationrate):
    with tf.variable_scope(scopename) as sc:
        x0 = layers.conv2d(bottom, filters, 1, padding='SAME', scope=scopename + '_conv0', activation_fn=None)
        x = layers.conv2d(bottom, filters, 1, padding='SAME', scope=scopename + '_conv1', activation_fn=tf.nn.relu)
        x = layers.conv2d(x, filters, 3, padding='SAME', rate=dilationrate, scope=scopename + '_conv2', activation_fn=tf.nn.relu)
        x = layers.conv2d(x, filters, 1, padding='SAME', scope=scopename + '_conv3', activation_fn=None)

        return tf.nn.relu(x+x0)

def _feature_extraction(bottom):
    with tf.variable_scope('FeatureExtraction') as sc:
        conv1_1=_resnet_layer(bottom, Feature_layer1, 'conv1_1',  1)
        conv1_2 = _resnet_layer(conv1_1, Feature_layer1, 'conv1_2',  1)
        pool1=layers.max_pool2d(conv1_2, 2, stride=2, scope='pool1')

        conv2_1 = _resnet_layer(pool1, Feature_layer2, 'conv2_1',  1)
        conv2_2 = _resnet_layer(conv2_1, Feature_layer2, 'conv2_2',  1)
        pool2 = layers.max_pool2d(conv2_2, 2, stride=2, scope='pool2')

        conv3_1 = _resnet_layer(pool2, Feature_layer3, 'conv3_1',  1)
        conv3_2 = _resnet_layer(conv3_1, Feature_layer3, 'conv3_2',  1)
        pool3 = layers.max_pool2d(conv3_2, 2, stride=2, scope='pool3')

        conv4_1 = _resnet_layer(pool3, Feature_layer4, 'conv4_1',  1)
        conv4_2 = _resnet_layer(conv4_1, Feature_layer4, 'conv4_2',  1)

        conv5_1 = _resnet_layer(conv4_2, Feature_layer5, 'conv5_1', 2)
        conv5_2 = _resnet_layer(conv5_1, Feature_layer5, 'conv5_2', 2)

        conv6_1 = _resnet_layer(conv5_2, Feature_layer6, 'conv6_1', 4)
        conv6_2 = _resnet_layer(conv6_1, Feature_layer6, 'conv6_2', 4)

        return conv6_2

def _upscore_layer(bottom, num_classes, scopename, ksize=4, stride=2):

    strides = [1, stride, stride, 1]
    in_features = bottom.shape[3]

    output_shape = tf.stack(
        [bottom.shape[0], 2*(bottom.shape[1]), 2*(bottom.shape[2]),
         num_classes])
    f_shape = [ksize, ksize, num_classes, in_features]
    weights = get_deconv_filter(f_shape, scopename)

    deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,
                                    strides=strides, padding='SAME')

    return deconv

def get_deconv_filter(f_shape, scopename):
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights,
                                   dtype=tf.float32)
    return tf.get_variable(name=scopename, initializer=init,
                           shape=weights.shape)


def _interpolation(bottom):
    with tf.variable_scope('Interpolation') as sc:
        conv6_1R = _resnet_layer(bottom, Feature_layer6, 'conv6_1R',  4)
        conv6_2R = _resnet_layer(conv6_1R, Feature_layer6, 'conv6_2R',  4)

        conv5_1R =_resnet_layer(conv6_2R, Feature_layer5, 'conv5_1R',  2)
        conv5_2R = _resnet_layer(conv5_1R, Feature_layer5, 'conv5_2R',  2)

        conv4_1R = _resnet_layer(conv5_2R, Feature_layer4, 'conv4_1R',  1)
        conv4_2R = _resnet_layer(conv4_1R, Feature_layer4, 'conv4_2R',  1)
        conv4_2RU = _upscore_layer(conv4_2R, Feature_layer4, 'conv4_2RU')

        conv3_1R = _resnet_layer(conv4_2RU, Feature_layer3, 'conv3_1R',  1)
        conv3_2R = _resnet_layer(conv3_1R, Feature_layer3, 'conv3_2R',  1)
        conv3_2RU = _upscore_layer(conv3_2R, Feature_layer3, 'conv3_2RU')

        conv2_1R = _resnet_layer(conv3_2RU, Feature_layer2, 'conv2_1R',  1)
        conv2_2R = _resnet_layer(conv2_1R, Feature_layer2, 'conv2_2R',  1)
        conv2_2RU = _upscore_layer(conv2_2R, Feature_layer2, 'conv2_2RU')

        conv1_1R = _resnet_layer(conv2_2RU, Feature_layer1, 'conv1_1R',  1)
        conv1_2R = _resnet_layer(conv1_1R, Feature_layer1, 'conv1_2R',  1)


        return conv1_2R






device = '/gpu:0'

with tf.device(device):
    print(tf.__version__)
    x0 = tf.placeholder(tf.float32,shape=(batch_size, inputdims, inputdims, 1))  # Input image
    y0 = tf.placeholder(tf.int32, shape=(batch_size, inputdims, inputdims)) # Output label
    weightmap = tf.placeholder(tf.float32, shape=(batch_size, inputdims, inputdims))
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    coarse_features = _feature_extraction(x0) # Generate features from the input image
    interpolated_output = _interpolation(coarse_features) # Upsampling the features to the size of the input image
    upscore0 = layers.conv2d(interpolated_output,num_classes, 1, padding='SAME', scope='score', activation_fn=None) # Predict the labels


    # logits0 = tf.reshape(upscore0, (-1, 1, 2))
    cross_entropy0 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=upscore0, labels=y0) # Cross-entropy



    #Loss calculation
    loss=tf.reduce_mean(cross_entropy0*weightmap)

    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)


load = 0 # 0 for fresh start and 1 for resume training from a previous optimization

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    tf.global_variables_initializer().run()
    saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=1)
    if load:
        path = tf.train.get_checkpoint_state('Checkpoints/')
        saver.restore(sess, path.model_checkpoint_path)
        step_start = int(path.model_checkpoint_path.split('-')[-1]) + 1
    else:
        step_start = 0

    print(step_start)
    totaliter=1000000 # Total epochs



    total_lossval=0.0
    loss_flag=0






    for i in range(step_start, totaliter):
        if (i<=10000):
            learning_ratef=initial_learning_rate

        if (i<=20000 and i>10000):
            learning_ratef=initial_learning_rate/10

        if (i<=30000 and i>20000):
            learning_ratef=initial_learning_rate/100

        if (i>30000):
            learning_ratef=initial_learning_rate/1000

        TrainingImages = os.listdir('TrainingImages')  # Directory for training images
        totalimages = len(TrainingImages)

        inputimages=np.zeros((batch_size,inputdims,inputdims,1))
        outputlables=np.zeros((batch_size,inputdims,inputdims))
        weightmatrix=np.zeros((batch_size,inputdims,inputdims))


        for batch_index in range(batch_size):
            randomindex = np.random.randint(0, high=totalimages)
            im = Image.open('TrainingImages/' + TrainingImages[randomindex])
            image = np.array(im, dtype=np.float32)
            image = image - 127.0
            image = image / 127.0  # Normalize the image between -1 to 1
            image = np.expand_dims(image, axis=2)
            inputimages[batch_index,:,:,:]=image
            label = scipy.io.loadmat('TrainingLabels/' + TrainingImages[randomindex][0:-4] + '.mat')['label']
            label = np.array(label, dtype=np.int32)
            outputlables[batch_index, :, :] = label
            edges = cv2.Canny(np.array(label*255, dtype=np.uint8),100,200)
            edges1 = np.array(edges, dtype=np.float32)
            im1 = cv2.GaussianBlur(edges1, (15, 15), 4)
            for _ in range(4):
                im1 = cv2.GaussianBlur(im1, (15, 15), 4)
            im1 = im1 / np.max(im1.flatten())
            im1 = weightfactor * im1 + 1  # weightfactor for the amplitude where you want to provide more weight
            where_are_NaNs = np.isnan(im1)
            im1[where_are_NaNs] = 1
            weightmatrix[batch_index,:,:]=im1

            # plt.figure(1)
            # plt.subplot(3,1,1)
            # plt.imshow(image[:,:,0],cmap='gray')
            # plt.subplot(3, 1, 2)
            # plt.imshow(label, cmap='gray')
            # plt.subplot(3, 1, 3)
            # plt.imshow(im1, cmap='gray')
            # plt.show()



        _, total_lossval1 = sess.run([train_step, loss], feed_dict={x0: inputimages, y0: outputlables, weightmap: weightmatrix,
                                                                    learning_rate: learning_ratef})
        total_lossval = total_lossval + total_lossval1
        loss_flag = loss_flag + 1
        # print(total_lossval1)

        if i % 1000 == 0:
            print(i, total_lossval / loss_flag, ',', learning_ratef)
            total_lossval = 0.0
            loss_flag = 0
            saver.save(sess, 'Checkpoints/' + 'model', global_step=i)

            #   Validation






