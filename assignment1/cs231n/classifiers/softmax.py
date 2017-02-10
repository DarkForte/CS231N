import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  num_train = X.shape[0]

  for i in xrange(num_train):
    now_train = X[i]
    f = np.dot(now_train, W)
    f -= np.max(f)
    p = np.exp(f[y[i]]) / np.sum(np.exp(f))
    loss += -np.log(p)

    for j in xrange(len(f)):
      pj = np.exp(f[j]) / np.sum(np.exp(f))
      dL_dF = pj

      if j==y[i]:
        dL_dF -= 1

      dW[:, j] += now_train.T * dL_dF

  dW /= num_train
  loss /= num_train
  loss += 0.5 * reg * np.sum(np.square(W))
  dW += reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  F = np.dot(X, W)
  F -= np.max(F, axis=1)[:, np.newaxis]
  row_sum = np.sum(np.exp(F), axis=1) #N
  P = np.exp(F[np.arange(num_train), y]) / row_sum
  loss = np.sum(-np.log(P)) / num_train
  loss += 0.5 * reg * np.sum(np.square(W))

  P_all = np.exp(F) / row_sum[:, np.newaxis]
  factors = np.zeros(F.shape) # NxC
  factors[np.arange(num_train), y] = -1
  dL_dF = P_all + factors
  dF_dW = X.T

  dW = np.dot(dF_dW, dL_dF)
  dW = dW / num_train + reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

