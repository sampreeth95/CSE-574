import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
from scipy.io import whosmat
import time
import pickle


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    sig = 1.0/(1.0 + np.exp(-1.0*z))
    return  sig# your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""
    
    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    labels_train = np.zeros((50000,1))
    labels_validation = np.zeros((10000,1))
    labels_test = np.zeros((10000,1))
    total_train = np.zeros((50000,784))
    validation_train = np.zeros((10000,784))
    total_test = np.zeros((10000,784))
    itr = 0
    ite = 0
    val = 0
    l = whosmat('mnist_all.mat')                                           # properties of data in .mat file
    for i in range(len(l)):
        data_name =l[i][0]                                                 # class name
        data_len = l[i][1][0]                                              # num of rows in each class
        data_class = data_name[-1]                                         # class identifier
        if "train" in data_name:
            train_perm = np.random.permutation(mat.get(data_name))         # randomly permute data
            labels_train[itr:(itr+data_len-1000),0] = data_class           # labels for train data
            total_train[itr:(itr+data_len-1000),:] = train_perm[1000:data_len,:] # train data
            labels_validation[val:val+1000,:] = data_class                   # labels for validation data
            validation_train[val:val+1000,:] = train_perm[0:1000,:]           # validation data (1000 from each class)
            itr = itr + data_len - 1000
            val = val + 1000
        else: 
            labels_test[ite:(ite+data_len),0] = data_class                  # labels for test data
            total_test[ite:(ite+data_len),:]=mat.get(data_name)              # test data
            ite=ite+data_len
    
    rand_num_t = np.random.permutation(np.arange(0,50000))
    train_data = total_train[rand_num_t,:]
    train_data = np.double(train_data)/255.0
    train_label = labels_train[rand_num_t,0]


    rand_num_v = np.random.permutation(np.arange(0,10000))
    validation_data = validation_train[rand_num_v,:]
    validation_data = np.double(validation_data)/255.0
    validation_label = labels_validation[rand_num_v,0]


    rand_num_te = np.random.permutation(np.arange(0,10000))
    test_data = total_test[rand_num_te,:]
    test_data = np.double(test_data)/255.0
    test_label = labels_test[rand_num_te,0]
    
    # Feature selection
    # Your code here.
    samefeatures = np.all(train_data == train_data[0,:],axis=0)
    selected_features = np.where(samefeatures==False)
#     print("selected features")
#     print(selected_features)
    train_data = train_data[:,samefeatures==False]
    validation_data=validation_data[:,samefeatures==False]
    test_data = test_data[:,samefeatures==False]
    print("Selected Features:",len(train_data[0]))
    
    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    n = training_data.shape[0]
    y = np.zeros((n, n_class))
    for i in range(n):
        index = int(training_label[i])
        y[i,index] = 1.0
    
    ip_ones = np.ones((n,1))                      # column vector of ones
    ip_hl = np.append(training_data, ip_ones, axis=1)      # adding bias to input data
    wmult_hl = np.dot(ip_hl,np.transpose(w1))     # multiplying weight vectors with input data
    op_hl = sigmoid(wmult_hl)                     # applying sigmoid activation fuction to get o/p of hidden layer
    hl_ones = np.ones((op_hl.shape[0],1))         # column vector of ones
    ip_ol = np.append(op_hl,hl_ones, axis=1)      # adding bias to o/p of hidden layer
    wmult_ol = np.dot(ip_ol, np.transpose(w2))    # multiplying weight vectors with o/p of hidden layer
    o = sigmoid(wmult_ol)                         # applying sigmoid activation fuction to get o/p

    error_fn = (-1/n)*(np.sum(np.multiply(y, np.log(o)) + np.multiply((1 - y), np.log((1 - o))))) # -ve logerror function
    regularization = (lambdaval / (2 * n)) * (np.sum(np.square(w1)) + np.sum(np.square(w2)))#regularization
    obj_val = error_fn + regularization           # error function with regularization
    
    del_l = np.subtract(o, y)                     # ol - yl
    grad_w2 = (1/n)*((np.dot(np.transpose(del_l), ip_ol)) + (lambdaval)*w2)  # (1/n)(deltal*zj + lambda*wlj)
    
    mult_del_l_w2 = np.dot(del_l, np.delete(w2, w2.shape[1]-1, 1))  # (sigma(deltalwlj)) 
    der_op_hl = np.multiply((1-op_hl),op_hl)                      # zj(1-zj) 
    grad_w1 = (1/n)*((np.dot(np.transpose(np.multiply(der_op_hl,mult_del_l_w2)), ip_hl)) + (lambdaval)*w1)
                                                                  # (1/n)*(zj(1-zj)*(sigma(deltalwlj))*xp + lambda*wjp)
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    
    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
#     obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here
    ip_ones = np.ones((data.shape[0],1))          # column vector of ones
    ip_hl = np.append(data, ip_ones, axis=1)      # adding bias to input data
    wmult_hl = np.dot(ip_hl,np.transpose(w1))     # multiplying weight vectors with input data
    op_hl = sigmoid(wmult_hl)                     # applying sigmoid activation fuction to get o/p of hidden layer
    hl_ones = np.ones((op_hl.shape[0],1))         # column vector of ones
    ip_ol = np.append(op_hl,hl_ones, axis=1)      # adding bias to o/p of hidden layer
    wmult_ol = np.dot(ip_ol, np.transpose(w2))    # multiplying weight vectors with o/p of hidden layer
    op_ol = sigmoid(wmult_ol)                     # applying sigmoid activation fuction to get o/p
#     op_ol_int = op_ol.astype(np.uint8)
    labels = op_ol.argmax(axis=1)           # returning class which has maximum value
    
    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features = preprocess()

#  Train Neural Network
# for i in range(4,29,4):
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50
print ("n_hidden: ", n_hidden)
# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

#     for j in range(0,61,5):
# set the regularization hyper-parameter
lambdaval = 10
print("lambda value: ",lambdaval)

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

T1 = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)
T2 = time.time()

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

print("Training Time: ", T2-T1)

obj = [selected_features, n_hidden, w1, w2, lambdaval]
print(obj)
pickle.dump(obj, open('params.pickle', 'wb'))
