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
keep_prob = tf.placeholder(tf.float32)

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



layer_conv21, weights_conv21 = \
    new_conv_layer(input=layer_conv12,
                   num_input_channels=num_filters12,
                   filter_size=filter_size21,
                   num_filters=num_filters21,
                   use_pooling=False)

layer_conv22, weights_conv22 = \
    new_conv_layer(input=layer_conv21,
                   num_input_channels=num_filters21,
                   filter_size=filter_size22,
                   num_filters=num_filters22,
                   use_pooling=True)



layer_conv31, weights_conv31 = \
    new_conv_layer(input=layer_conv22,
                   num_input_channels=num_filters22,
                   filter_size=filter_size31,
                   num_filters=num_filters31,
                   use_pooling=False)

layer_conv32, weights_conv32 = \
    new_conv_layer(input=layer_conv31,
                   num_input_channels=num_filters31,
                   filter_size=filter_size32,
                   num_filters=num_filters32,
                   use_pooling=True)



layer_conv41, weights_conv41 = \
    new_conv_layer(input=layer_conv32,
                   num_input_channels=num_filters32,
                   filter_size=filter_size41,
                   num_filters=num_filters41,
                   use_pooling=False)

layer_conv42, weights_conv42 = \
    new_conv_layer(input=layer_conv41,
                   num_input_channels=num_filters41,
                   filter_size=filter_size42,
                   num_filters=num_filters42,
                   use_pooling=True)



layer_conv51, weights_conv51 = \
    new_conv_layer(input=layer_conv42,
                   num_input_channels=num_filters42,
                   filter_size=filter_size51,
                   num_filters=num_filters51,
                   use_pooling=False)

layer_conv52, weights_conv52 = \
    new_conv_layer(input=layer_conv51,
                   num_input_channels=num_filters51,
                   filter_size=filter_size52,
                   num_filters=num_filters52,
                   use_pooling=False)

layer_pool5 = \
   new_avg_pool_layer(input=layer_conv52)

layer_dropout = tf.nn.dropout(layer_pool5, keep_prob)

layer_flat, num_features = flatten_layer(layer_dropout)


layer_fc1 = new_fc_layer(input=layer_flat,
                         num_inputs=num_features,
                         num_outputs=n_classes,
                         use_relu=False)


#Prediction
y_pred = tf.nn.softmax(layer_fc1)
y_pred_cls = tf.argmax(y_pred, dimension=1)


#Cost Function and optimizer
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc1,
                                                        labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)

#Accuracy
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
