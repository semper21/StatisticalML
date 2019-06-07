import numpy as np
from random import shuffle
import scipy.sparse
from IPython import embed

  
def softmax_loss_vectorized(theta, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    
    J = 0.0
    grad = np.zeros_like(theta)
    #X = X.reshape(66257, 3073)
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
    try:
        J = -1./m * np.sum(I * np.log(prob)) + reg/(2.*m) * np.sum(theta ** 2)
    except ValueError:
        embed()
    grad = -1./m * (I-prob).dot(X).T + 1.*reg/m*theta
    
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################
    
    return J, grad
