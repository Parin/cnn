import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *



class ConvNet(object):
  """
  A convolutional network with the following architecture:
  
  INPUT -> [[CONV -> RELU]*X -> POOL?]*Y -> [FC -> RELU]*Z -> FC
  X = 2 with POOL Y = 5 and Z = 3
  
  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """
  
  def __init__(self, 
               input_dim=(3, 32, 32),  
               filter_size=3,
               hidden_dim=100, 
               num_classes=10, 
               weight_scale=1e-2, 
               reg=0.0,
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
    self.bn_params = []
    self.reg = reg
    self.dtype = dtype
        
    X, Y, Z = 2, 5, 3
    ############################################################################
    # Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    '''
    K = [0, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512] # 5 * 2 conv layers
    F = filter_size
    S = 1
    P = (F-1)/2
        
    D1, W1, H1 = input_dim
    for i in xrange(1, 11):
        
        self.params['W' + str(i)] = np.random.randn(K[i], D1, F, F).astype(dtype) * weight_scale
        self.params['b' + str(i)] = np.ones(K[i], dtype=dtype)
    
        D2 = K[i]
        W2 = 1 + (W1 + 2 * P - F) / S
        H2 = 1 + (H1 + 2 * P - F) / S
    
        self.params['gamma' + str(i)] = np.ones(D2)
        self.params['beta' + str(i)] = np.zeros(D2)
        self.bn_params.append({'mode': 'train'})
    
        print "W{0} : {1}, {2}".format(i, self.params['W' + str(i)].shape, np.prod(self.params['W' + str(i)].shape))
        print "b{0} : {1}".format(i, self.params['b' + str(i)].shape)
        print "gamma{0} : {1}".format(i, self.params['gamma' + str(i)].shape)
        print "beta{0} : {1}".format(i, self.params['beta'+ str(i)].shape)
        
        D1, W1, H1 = D2, W2, H2 
    '''
    K = [0, 32, 32, 64, 64, 128, 128, 256, 256, 512, 512] # 5 * 2 conv layers
    F = filter_size
    S = 1
    P = (F-1)/2
        
    D1, W1, H1 = input_dim
    
    lines = open('./net.txt', 'r').readlines()
    for line in lines:
        
        tokens = line.split('\t')
        
        if tokens[0] == 'conv':
            
            id = tokens[1]
            K = int(tokens[2])
            
            self.params['W' + id] = np.random.randn(K, D1, F, F).astype(dtype) * weight_scale
            self.params['b' + id] = np.ones(K, dtype=dtype)
            
            D2 = K
            W2 = 1 + (W1 + 2 * P - F) / S
            H2 = 1 + (H1 + 2 * P - F) / S
            
            D1, W1, H1 = D2, W2, H2
            
            print "W{0} : {1}, {2}".format(i, self.params['W' + id].shape, np.prod(self.params['W' + id].shape))
            print "b{0} : {1}".format(i, self.params['b' + id].shape)

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
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    
    # pass conv_param to the forward pass for the convolutional layer
    #filter_size = W1.shape[2]
    #conv_param = {'stride': 1, 'pad': (filter_size - 1) / 2}

    # pass pool_param to the forward pass for the max-pooling layer
    #pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    
    a1, conv_cache = conv_relu_pool_forward(X, W1, b1, self.conv_param, self.pool_param)
    a2, hidden_cache = affine_relu_forward(a1, W2, b2)
    scores, afine_cache = affine_forward(a2, W3, b3)
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    if y is None:
      return scores
    
    loss, grads = 0, {}
    ############################################################################
    # Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    
    loss, dscores = softmax_loss(scores, y)
    da2, grads['W3'], grads['b3'] = affine_backward(dscores, afine_cache)
    da1, grads['W2'], grads['b2'] = affine_relu_backward(da2, hidden_cache)
    dx, grads['W1'], grads['b1'] = conv_relu_pool_backward(da1, conv_cache)
    
    # regularization
    #loss += 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    reg_loss = 0.5 * self.reg * (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    #print (np.sum(W1 * W1) + np.sum(W2 * W2) + np.sum(W3 * W3))
    loss += reg_loss
    grads['W3'] += self.reg * W3
    grads['W2'] += self.reg * W2
    grads['W1'] += self.reg * W1
    
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################
    
    return loss, grads
  
  
pass
