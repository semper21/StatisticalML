import numpy as np
from random import shuffle
import scipy.sparse

def softmax_loss_naive(theta, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)
  Inputs:
  - theta: d x K parameter matrix. Each column is a coefficient vector for class k
  - X: m x d array of data. Data are d-dimensional rows.
  - y: 1-dimensional array of length m with labels 0...K-1, for K classes
  - reg: (float) regularization strength
  Returns:
  a tuple of:
  - loss as single float
  - gradient with respect to parameter matrix theta, an array of same size as theta
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in J and the gradient in grad. If you are not              #
  # careful here, it is easy to run into numeric instability. Don't forget    #
  # the regularization term!                                                  #
  #############################################################################
  for i in xrange(m):
        for k in xrange(theta.shape[1]):
            t = X[i].dot(theta)
            denom = sum(np.exp(t-max(t)))
            prob = np.exp(X[i].dot(theta.T[k])-max(t)) / denom
            I = 1 if y[i] == k else 0
            J += -1./m * I * np.log(prob)
            grad.T[k] += -1./m * X[i]* (I - prob)
  J += reg/(2.*m) * np.sum(theta ** 2)
  grad += 1.*reg/m * theta
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad

  
def softmax_loss_vectorized(theta, X, y, reg):
  """
  Softmax loss function, vectorized version.
  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.

  J = 0.0
  grad = np.zeros_like(theta)
  m, dim = X.shape

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in J and the gradient in grad. If you are not careful      #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization term!                                                      #
  #############################################################################

  mat = np.indices((m, 10))[1]
  I = (mat.T==y)*1.0
  t = X.dot(theta)
  p = np.exp(t.T - np.max(t, axis=1)) 
  prob = p/np.sum(p, axis=0)
  J = -1./m * np.sum(I * np.log(prob)) + reg/(2.*m) * np.sum(theta ** 2)
  
  grad = -1./m * (I-prob).dot(X).T + 1.*reg/m*theta

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return J, grad
