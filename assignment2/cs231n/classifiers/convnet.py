import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ConvNet(object):
  """
  A multi-layer convolutional network with the following architecture:

  [conv - relu - 2x2 max pool]x2 - [affine - relu]x2 - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=(32, 8), filter_size=(7, 3),
               hidden_dim=(100, 30), num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.num_filters = num_filters
    self.filter_size = filter_size

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    C, H, W = input_dim
    self.conv_layers = len(num_filters)
    self.params["W_conv0"] = np.random.normal(0, weight_scale, (num_filters[0], input_dim[0], filter_size[0], filter_size[0]))
    self.params["b_conv0"] = np.zeros(num_filters[0])

    for i in range(1, self.conv_layers):
      self.params["W_conv%d" % i] = np.random.normal(0, weight_scale, (num_filters[i], num_filters[i-1], filter_size[i], filter_size[i]))
      self.params["b_conv%d" % i] = np.zeros(num_filters[i])

    self.fc_layers = len(hidden_dim)
    self.params["W_fc0"] = np.random.normal(0, weight_scale, (num_filters[-1]*H*W/(4**self.conv_layers), hidden_dim[0]))
    self.params["b_fc0"] = np.zeros(hidden_dim[0])
    for i in range(1, self.fc_layers):
      self.params["W_fc%d" % i] = np.random.normal(0, weight_scale, (hidden_dim[i-1], hidden_dim[i]))
      self.params["b_fc%d" % i] = np.zeros(hidden_dim[i])

    self.params["W_soft"] = np.random.normal(0, weight_scale, (hidden_dim[-1], num_classes))
    self.params["b_soft"] = np.zeros(num_classes)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.iteritems():
      self.params[k] = v.astype(dtype)

  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """

    # pass conv_param to the forward pass for the convolutional layer
    conv_param = [{'stride': 1, 'pad': (now_filter_size - 1) / 2} for now_filter_size in self.filter_size]

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    conv_caches = []
    out = X
    for i in range(self.conv_layers):
      out, cache = conv_relu_pool_forward(out, self.params["W_conv%d" % i], self.params["b_conv%d" % i], conv_param[i], pool_param)
      conv_caches.append(cache)

    fc_caches = []
    for i in range(self.fc_layers):
      out, cache = affine_relu_forward(out, self.params["W_fc%d" % i] , self.params["b_fc%d" %i])
      fc_caches.append(cache)

    out, softmax_cache = affine_forward(out, self.params["W_soft"], self.params["b_soft"])

    scores = out

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    loss, dx = softmax_loss(scores, y)
    regloss = 0
    for i in range(self.conv_layers):
      regloss += np.sum(self.params["W_conv%d" % i] ** 2)
    for i in range(self.fc_layers):
      regloss += np.sum(self.params["W_fc%d" % i] ** 2)
    regloss += np.sum(self.params["W_soft"]**2)

    loss += 0.5 * self.reg * regloss

    dx, grads["W_soft"], grads["b_soft"] = affine_backward(dx, softmax_cache)
    grads["W_soft"] += self.reg * self.params["W_soft"]

    for i in range(self.fc_layers-1, -1, -1):
      dx, grads["W_fc%d" % i], grads["b_fc%d" % i] = affine_relu_backward(dx, fc_caches[i])
      grads["W_fc%d" % i] += self.reg * self.params["W_fc%d" % i]

    for i in range(self.conv_layers-1, -1, -1):
      dx, grads["W_conv%d" %i], grads["b_conv%d" %i] = conv_relu_pool_backward(dx, conv_caches[i])
      grads["W_conv%d" % i] += self.reg * self.params["W_conv%d" % i]

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


pass
