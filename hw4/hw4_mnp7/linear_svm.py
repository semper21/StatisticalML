import numpy as np

##################################################################################
#   Two class or binary SVM                                                      #
##################################################################################

def binary_svm_loss(theta, X, y, C):
  """
  SVM hinge loss function for two class problem

  Inputs:
  - theta: A numpy vector of size d containing coefficients.
  - X: A numpy array of shape mxd 
  - y: A numpy array of shape (m,) containing training labels; +1, -1
  - C: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to theta; an array of same shape as theta
"""

  m, d = X.shape
  grad = np.zeros(theta.shape)
  J = 0

  ############################################################################
  # TODO                                                                     #
  # Implement the binary SVM hinge loss function here                        #
  # 4 - 5 lines of vectorized code expected                                  #
  ############################################################################
  correctness = y * X.dot(theta)
  J = 1./(2*m) * sum(theta**2) + 1.*C/m * sum(1-correctness[correctness<1])
  grad = 1./m * theta + 1.*C/m * -y[correctness<1].dot(X[correctness<1])

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, grad

##################################################################################
#   Multiclass SVM                                                               #
##################################################################################

# SVM multiclass

def svm_loss_naive(theta, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension d, there are K classes, and we operate on minibatches
  of m examples.

  Inputs:
  - theta: A numpy array of shape d X K containing parameters.
  - X: A numpy array of shape m X d containing a minibatch of data.
  - y: A numpy array of shape (m,) containing training labels; y[i] = k means
    that X[i] has label k, where 0 <= k < K.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss J as single float
  - gradient with respect to weights theta; an array of same shape as theta
  """

  delta = 1.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero

  # compute the loss function

  K = theta.shape[1]
  m = X.shape[0]
  J = 0.0
  for i in xrange(m):
    scores = X[i,:].dot(theta)
    correct_class_score = scores[y[i]]
    for j in xrange(K):
      if j == y[i]:
        continue
      margin = max(0,scores[j] - correct_class_score + delta)
      J += margin
      p = scores[j] - correct_class_score + delta
      if p > 0:
          dtheta[:,j] += X[i]
          dtheta[:,y[i]] -= X[i]

  # Right now the loss is a sum over all training examples, but we want it
  # To be an average instead so we divide by num_train.
  J /= m
  dtheta /= m

  # Add regularization to the loss.
  J += 0.5 * reg * np.sum(theta * theta)
  dtheta += reg * theta
 
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dtheta.            #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
    
  return J, dtheta


def svm_loss_vectorized(theta, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  J = 0.0
  dtheta = np.zeros(theta.shape) # initialize the gradient as zero
  delta = 1.0
  m = X.shape[0]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in variable J.                                                     #
  #############################################################################
  
  scores = X.dot(theta)
  #print scores[np.arange(m),y].shape
  margin = np.maximum(0, (scores.T - scores[np.arange(m),y]).T + delta)
  margin[np.arange(m),y]=0
  J = 1./m * np.sum(margin) + 0.5*reg*np.sum(theta.T.dot(theta))

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  #This mask flags the examples whose margin is greater than 0
  X_mask = np.zeros(margin.shape)
  X_mask[margin>0] = 1
  
  #count number of examples with margin>0
  c = np.sum(X_mask,axis=1)
  X_mask[np.arange(m),y] = -c
  
  dtheta = X.T.dot(X_mask)
  dtheta /= m
  dtheta += reg * theta


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dtheta.                                       #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return J, dtheta
