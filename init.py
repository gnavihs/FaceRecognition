#matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

exec(open("./model.py").read())


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true = tf.placeholder(tf.float32, shape=[None, n_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

layer_conv11, weights_conv11 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size11,
                   num_filters=num_filters11,
                   use_pooling=False)



layer_conv12, weights_conv12 = \
    new_conv_layer(input=layer_conv11,
                   num_input_channels=num_filters11,
                   filter_size=filter_size12,
                   num_filters=num_filters12,
                   use_pooling=True)

layer_conv11, weights_conv11 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size11,
                   num_filters=num_filters11,
                   use_pooling=False)



layer_conv12, weights_conv12 = \
    new_conv_layer(input=layer_conv11,
                   num_input_channels=num_filters11,
                   filter_size=filter_size12,
                   num_filters=num_filters12,
                   use_pooling=True)

layer_conv11, weights_conv11 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size11,
                   num_filters=num_filters11,
                   use_pooling=False)



layer_conv12, weights_conv12 = \
    new_conv_layer(input=layer_conv11,
                   num_input_channels=num_filters11,
                   filter_size=filter_size12,
                   num_filters=num_filters12,
                   use_pooling=True)

layer_conv11, weights_conv11 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size11,
                   num_filters=num_filters11,
                   use_pooling=False)



layer_conv12, weights_conv12 = \
    new_conv_layer(input=layer_conv11,
                   num_input_channels=num_filters11,
                   filter_size=filter_size12,
                   num_filters=num_filters12,
                   use_pooling=True)

layer_conv11, weights_conv11 = \
    new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size11,
                   num_filters=num_filters11,
                   use_pooling=False)



layer_conv12, weights_conv12 = \
    new_conv_layer(input=layer_conv11,
                   num_input_channels=num_filters11,
                   filter_size=filter_size12,
                   num_filters=num_filters12,
                   use_pooling=True)



layer_flat, num_features = flatten_layer(layer_conv12)


layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=fc_size,
                         use_relu=True)




layer_fc2 = new_fc_layer(input=layer_fc1,
                         num_inputs=fc_size,
                         num_outputs=n_classes,
                         use_relu=False)



y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)


#Cost Function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

