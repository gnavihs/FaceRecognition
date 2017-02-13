
# Convolutional Layer 11.
filter_size11 = 3          # Convolution filters are 3x3 pixels.
num_filters11 = 32         # There are 32 of these filters.

# Convolutional Layer 12.
filter_size12 = 3          # Convolution filters are 3x3 pixels.
num_filters12 = 64         # There are 64 of these filters.


# Convolutional Layer 21.
filter_size21 = 3          # Convolution filters are 3x3 pixels.
num_filters21 = 64         # There are 64 of these filters.

# Convolutional Layer 22.
filter_size22 = 3          # Convolution filters are 3x3 pixels.
num_filters22 = 128         # There are 128 of these filters.

# Convolutional Layer 31.
filter_size31 = 3          # Convolution filters are 3x3 pixels.
num_filters31 = 96         # There are 96 of these filters.

# Convolutional Layer 32.
filter_size32 = 3          # Convolution filters are 3x3 pixels.
num_filters32 = 192         # There are 192 of these filters.

# Convolutional Layer 41.
filter_size41 = 3          # Convolution filters are 3x3 pixels.
num_filters41 = 128         # There are 128 of these filters.

# Convolutional Layer 42.
filter_size42 = 3          # Convolution filters are 3x3 pixels.
num_filters42 = 256         # There are 256 of these filters.

# Convolutional Layer 51.
filter_size51 = 3          # Convolution filters are 3x3 pixels.
num_filters51 = 160         # There are 160 of these filters.

# Convolutional Layer 52.
filter_size52 = 3          # Convolution filters are 3x3 pixels.
num_filters52 = 320         # There are 320 of these filters.

# We know that MNIST images are 100 pixels in each dimension.
img_size = h

# Images are stored in one-dimensional arrays of this length.
img_size_flat = img_size * img_size

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# Number of colour channels for the images: 1 channel for gray-scale.
num_channels = 1


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):     # Use 2x2 max-pooling.
  shape = [filter_size, filter_size, num_input_channels, num_filters]

  # Create new weights aka. filters with the given shape.
  weights = new_weights(shape=shape)

  # Create new biases, one for each filter.
  biases = new_biases(length=num_filters)

  layer = tf.nn.conv2d(input=input,
                       filter=weights,
                       strides=[1, 1, 1, 1],
                       padding='SAME')

  layer += biases

  # Use pooling to down-sample the image resolution?
  if use_pooling:
      layer = tf.nn.max_pool(value=layer,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME')

  layer = tf.nn.relu(layer)
  return layer, weights


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()  
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    return layer_flat, num_features

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

def new_avg_pool_layer(input):
  layer = tf.nn.avg_pool(value=input,
                         ksize=[1, 7, 7, 1],
                         strides=[1, 2, 2, 1],
                         padding='VALID')
  return layer

def next_batch(aDataset, batch_size, shuffle=True):
  # Ensure we update the global variable rather than a local copy.
  global _epochs_completed
  global _index_in_epoch
  global _images
  global _labels
  _num_examples = aDataset.images.shape[0]
  start = _index_in_epoch
  
  if _epochs_completed == 0 and start == 0 and shuffle:
    perm0 = np.arange(_num_examples)
    np.random.shuffle(perm0)
    _images = aDataset.images[perm0]
    _labels = aDataset.labels[perm0]
  # Go to the next epoch
  if start + batch_size > _num_examples:
    # Finished epoch
    _epochs_completed += 1
    # Get the rest examples in this epoch
    rest_num_examples = _num_examples - start
    images_rest_part = _images[start:_num_examples]
    labels_rest_part = _labels[start:_num_examples]
    # Shuffle the data
    if shuffle:
      perm = np.arange(_num_examples)
      np.random.shuffle(perm)
      _images = aDataset.images[perm]
      _labels = aDataset.labels[perm]
    # Start next epoch
    start = 0
    _index_in_epoch = batch_size - rest_num_examples
    end = _index_in_epoch
    images_new_part = _images[start:end]
    labels_new_part = _labels[start:end]
    return np.concatenate((images_rest_part, images_new_part), axis=0) , np.concatenate((labels_rest_part, labels_new_part), axis=0)
  else:
    _index_in_epoch += batch_size
    end = _index_in_epoch
    return _images[start:end], _labels[start:end]
