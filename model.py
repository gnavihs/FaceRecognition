
#data.test.cls = np.argmax(data.test.labels, axis=1)

# Convolutional Layer 11.
filter_size11 = 3          # Convolution filters are 3x3 pixels.
num_filters11 = 32         # There are 16 of these filters.

# Convolutional Layer 12.
filter_size12 = 3          # Convolution filters are 3x3 pixels.
num_filters12 = 32         # There are 36 of these filters.


# Convolutional Layer 21.
filter_size21 = 3          # Convolution filters are 3x3 pixels.
num_filters21 = 64         # There are 16 of these filters.

# Convolutional Layer 22.
filter_size22 = 3          # Convolution filters are 3x3 pixels.
num_filters22 = 128         # There are 36 of these filters.

# Convolutional Layer 31.
filter_size31 = 3          # Convolution filters are 3x3 pixels.
num_filters31 = 96         # There are 16 of these filters.

# Convolutional Layer 32.
filter_size32 = 3          # Convolution filters are 3x3 pixels.
num_filters32 = 192         # There are 36 of these filters.

# Convolutional Layer 41.
filter_size41 = 3          # Convolution filters are 3x3 pixels.
num_filters41 = 128         # There are 16 of these filters.

# Convolutional Layer 42.
filter_size42 = 3          # Convolution filters are 3x3 pixels.
num_filters42 = 256         # There are 36 of these filters.

# Convolutional Layer 51.
filter_size51 = 3          # Convolution filters are 3x3 pixels.
num_filters51 = 160         # There are 16 of these filters.

# Convolutional Layer 52.
filter_size52 = 3          # Convolution filters are 3x3 pixels.
num_filters52 = 320         # There are 36 of these filters.

# Fully-connected layer.
fc_size = 15               # Number of neurons in fully-connected layer.


# We know that MNIST images are 100 pixels in each dimension.
img_size = 10

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
                   use_pooling=True,   # Use 2x2 max-pooling.
                   avg_pooling=False): # Use 7x7 avg-pooling. 

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Use pooling to down-sample the image resolution?
    if avg_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.avg_pool(value=layer,
                               ksize=[1, 7, 7, 1],
                               strides=[1, 2, 2, 1],
                               padding='VALID')
    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights




def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
    # Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
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


def next_batch(aDataset, batch_size, shuffle=True):
  # Ensure we update the global variable rather than a local copy.
  global _epochs_completed
  global _index_in_epoch
  global _images
  global _labels
  _num_examples = aDataset.images.shape[0]
  start = _index_in_epoch
  if _epochs_completed == 0 and start == 0 and shuffle:
    perm0 = numpy.arange(_num_examples)
    numpy.random.shuffle(perm0)
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
      perm = numpy.arange(_num_examples)
      numpy.random.shuffle(perm)
      _images = aDataset.images[perm]
      _labels = aDataset.labels[perm]
    # Start next epoch
    start = 0
    _index_in_epoch = batch_size - rest_num_examples
    end = _index_in_epoch
    images_new_part = _images[start:end]
    labels_new_part = _labels[start:end]
    return numpy.concatenate((images_rest_part, images_new_part), axis=0) , numpy.concatenate((labels_rest_part, labels_new_part), axis=0)
  else:
    _index_in_epoch += batch_size
    end = _index_in_epoch
    return _images[start:end], _labels[start:end]
