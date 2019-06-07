#! /home/vision/anaconda2/bin/python
'''
Created on Apr 12, 2017

@author: ywkim

- ReLU neurons
- stride = 1
- zero padded
- max pooling over 2x2 blocks
- conv1: 32 filters (5x5)
- conv2: 64 filters (5x5)

'''



import csv
from csv import reader
from sys import argv
import numpy as np
import tensorflow as tf

# Load CSV file
def load_csv(filename):
    dataset = []
    labelset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader, None) #skip header
        for row in csv_reader:
            if not row:
                continue
            if row[0] == '10':
                label = 0
            else:
                #print row[0]
                label = int(row[0])
            data = row[1:]
            data = map(lambda x: float(x), data)
            dataset.append(data)
            labelset.append(label)
    X = np.array(dataset)
    X_train = X.reshape(32, 32, 3, -1).transpose(3,0,1,2)
    Y_train = np.array(labelset)
    return X_train, Y_train

def load_csv_test(filename):
    dataset = []
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        next(csv_reader, None) #skip header
        for row in csv_reader:
            if not row:
                continue
            row = map(lambda x: float(x), row)
            dataset.append(row)
    X = np.array(dataset)
    X_test = X.reshape(32, 32, 3, -1).transpose(3,0,1,2)
    #X_test = np.vsplit(X_test, 8)
    #split into 8 groups and return 
    return X_test

def convert2grayscale(numSample, X):
    #convert rgb->grayscale
    X_new = np.zeros((numSample, 32, 32))
    for i in range(numSample):
        image = X[i]
        gray = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                gray[rownum][colnum] = weightedAverage(image[rownum][colnum])

        #X[i] = np.array([gray]*3).reshape(32,32,3)
        X_new[i] = gray
    return X_new

def rgb2YUV(rgb):
    rgb2yuv = np.array([[0.299, 0.587, 0.114],
                   [-0.14713, -0.28886, 0.436],
                   [0.615, -0.51499, -0.10001]])
    return np.dot(rgb[...,:3], rgb2yuv.T)
        
def preprocess(X_train, X_val, X_test):
    # Preprocessing: reshape the image data into rows 
    
    original_Xtrain = X_train.shape
    original_Xval = X_val.shape
    original_Xtest = X_test.shape
    
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    
    # As a sanity check, print out the shapes of the data
    print 'Training data shape: ', X_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Test data shape: ', X_test.shape
    
    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    mean_image = np.mean(X_train, axis=0)
    std_image = np.std(X_train, axis=0)
    
    # second: subtract the mean image from train and test data
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # third: divide by std
    X_train = X_train/std_image
    X_val = X_val/std_image
    X_test = X_test/std_image
    
    
    X_train = X_train.reshape(original_Xtrain[0], original_Xtrain[1], original_Xtrain[2])
    X_val = X_val.reshape(original_Xval[0], original_Xval[1], original_Xval[2])
    X_test = X_test.reshape(original_Xtest[0], original_Xtest[1], original_Xtest[2])
    
    return X_train, X_val, X_test



def normalize3(X_train, X_val, X_test):
    # Preprocessing: reshape the image data into rows 
    
    original_Xtrain = X_train.shape
    original_Xval = X_val.shape
    original_Xtest = X_test.shape
    
    X_train = np.reshape(X_train, (X_train.shape[0], -1))
    X_val = np.reshape(X_val, (X_val.shape[0], -1))
    X_test = np.reshape(X_test, (X_test.shape[0], -1))
    
    # As a sanity check, print out the shapes of the data
    print 'Training data shape: ', X_train.shape
    print 'Validation data shape: ', X_val.shape
    print 'Test data shape: ', X_test.shape
    
    # Preprocessing: subtract the mean image
    # first: compute the image mean based on the training data
    
    min_Train = np.min(X_train, axis=1, keepdims = True)
    max_Train = np.max(X_train, axis=1, keepdims = True)
    range_Train = max_Train - min_Train
    
    min_Val = np.min(X_val, axis=1, keepdims = True)
    max_Val = np.max(X_val, axis=1, keepdims = True)
    range_Val = max_Val - min_Val
    
    min_Test = np.min(X_test, axis=1, keepdims = True)
    max_Test = np.max(X_test, axis=1, keepdims = True)
    range_Test = max_Test - min_Test
    
    # second: subtract the mean image from train and test data
    X_train -= min_Train
    X_val -= min_Val
    X_test -= min_Test
    
    # third: divide by std
    X_train = X_train/range_Train
    X_val = X_val/range_Val
    X_test = X_test/range_Test
    
    
    X_train = X_train.reshape(original_Xtrain[0], original_Xtrain[1], original_Xtrain[2])
    X_val = X_val.reshape(original_Xval[0], original_Xval[1], original_Xval[2])
    X_test = X_test.reshape(original_Xtest[0], original_Xtest[1], original_Xtest[2])
    
    return X_train, X_val, X_test


def subsample(num_training,num_validation,X_train,y_train):
    # Our validation set will be num_validation points from the original
    # training set.
    
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    Y_val = y_train[mask]
    
    # Our training set will be the first num_train points from the original
    # training set.
    mask = range(num_training)
    X_train = X_train[mask]
    Y_train = y_train[mask]
    
    return X_train, Y_train, X_val, Y_val

def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.589*pixel[1] + 0.114*pixel[2]  

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

def next_batch(arrayX, arrayY, sampSize):
    assert len(arrayX) == len(arrayY)
    shuffledX = np.empty((sampSize, 32, 32), dtype=arrayX.dtype)
    shuffledY = np.empty((sampSize, 10), dtype=arrayY.dtype)
    p = np.random.choice(len(X_train), sampSize, replace = False)
    for i in range(sampSize):
        shuffledX[i] = arrayX[p[i]]
        shuffledY[i] = arrayY[p[i]]
    
    return shuffledX, shuffledY

def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def nn_layer(input_tensor, input_dim, output_dim, layer_name):

    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable(input_dim)
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
            
        with tf.name_scope('activation'):
            activations = tf.nn.relu(conv2d(input_tensor, weights)+ biases)
        
        with tf.name_scope('pool'):
            pool = max_pool_2x2(activations)
            
        tf.summary.histogram('activations', activations)
        return pool


if __name__ == '__main__':
    #trainfile = argv[1]
    #testfile = argv[2]
    trainfile = '/home/vision/Documents/Kaggle_SVHN/train.csv'
    testfile= '/home/vision/Documents/Kaggle_SVHN/test.csv'
    X_train, Y_train = load_csv(trainfile)
    X_test = load_csv_test(testfile)

    print '---finished loading data---'
    '''
    X_train = rgb2YUV(X_train)
    X_test = rgb2YUV(X_test)

    '''
    print 'shape before conversion: ', X_train.shape
    
    X_train = convert2grayscale(73257, X_train)
    print 'shape after conversion: ', X_train.shape
    X_test = convert2grayscale(26032, X_test)
    
    X_train, Y_train, X_val, Y_val = subsample(55257, 18000, X_train, Y_train)

    #X_train, X_val, X_test = preprocess(X_train, X_val, X_test)
    X_train, X_val, X_test = normalize3(X_train, X_val, X_test)
        
    print '---finished preprocessing---'
    
    
    ##########################################
    # Training data shape:  (66257, 32, 32)  #
    # Validation data shape:  (7000, 32, 32) #
    # Test data shape:  (26032, 1024)        #
    ##########################################

    
    sess = tf.InteractiveSession()

    #implementing TF
    x = tf.placeholder(tf.float32, shape=[None, 32, 32]) #grayscaled, so (32x32x1)
    y_ = tf.placeholder(tf.float32, shape=[None, 10]) #[None, 10] for one-hot 10-dimensional vectors


    #(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu)
    x_image = tf.reshape(x, [-1, 32, 32, 1]) #1 = # of color channels
    conv1 = nn_layer(x_image, [5,5,1,32], 32, 'conv1')
    conv2 = nn_layer(conv1, [3,3,32, 64], 64, 'conv2')
    

    '''    
    #first layer (conv + max)
    W_conv1 = weight_variable([5, 5, 1, 32]) #change 32 to desired # of filters
    b_conv1 = bias_variable([32]) #change 32 to desired # of filters
    
    x_image = tf.reshape(x, [-1, 32, 32, 1]) #1 = # of color channels
    #convolve x_image with the weight tensor, add the bias, apply the ReLU function
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    #max_pool_2x2 method will reduce the image size to 16x16.
    h_pool1 = max_pool_2x2(h_conv1)
    
    #second layer (64 features for each 5x5 patch -> image size will be reduced to 8x8)
    W_conv2 = weight_variable([3, 3, 32, 64])
    b_conv2 = bias_variable([64])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    #third layer
    W_conv3 = weight_variable([3, 3, 64, 128])
    b_conv3 = bias_variable([128])
    
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    
    
    #fourth layer
    W_conv4 = weight_variable([3, 3, 128, 256])
    b_conv4 = bias_variable([256])
    
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)
    
    '''
    

    #fully-connected layer with 4096 neurons 
    with tf.name_scope('fully-connected'):
        with tf.name_scope('weights'):
            weights = weight_variable([8 * 8 * 64, 1024])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([1024])
            variable_summaries(biases)
        with tf.name_scope('activation'):
            #reshape the tensor from the pooling layer into a batch of vectors
            h_pool2_flat = tf.reshape(conv2, [-1, 8 * 8 * 64])
            #multiply by a weight matrix, add a bias, and apply a ReLU
            activations = tf.nn.relu(tf.matmul(h_pool2_flat, weights) + biases)
        tf.summary.histogram('activations', activations)
        
        
    '''        
    W_fc1 = weight_variable([8 * 8 * 64, 1024]) #might be better with 500
    b_fc1 = bias_variable([1024])
    #reshape the tensor from the pooling layer into a batch of vectors
    h_pool2_flat = tf.reshape(conv2, [-1, 8 * 8 * 64])
    #multiply by a weight matrix, add a bias, and apply a ReLU
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    #fully-connected layer with 4096 neurons 
    W_fc2 = weight_variable([4096, 4096]) 
    b_fc2 = bias_variable([4096])
    #multiply by a weight matrix, add a bias, and apply a ReLU
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

    #fully-connected layer with 1000 neurons 
    W_fc3 = weight_variable([4096, 1000]) 
    b_fc3 = bias_variable([1000])
    #multiply by a weight matrix, add a bias, and apply a ReLU
    h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)    
    '''
    
    #dropout
    with tf.name_scope('dropout'):
        keep_prob = tf.placeholder(tf.float32)
        tf.summary.scalar('dropout_keep_probability', keep_prob)
        h_fc1_drop = tf.nn.dropout(activations, keep_prob)
    
    
    with tf.name_scope('softmax'):
        #softmax (readout layer)
        W_fc4 = weight_variable([1024, 10]) #why 10?
        variable_summaries(W_fc4)
        b_fc4 = bias_variable([10]) #why 10?
        variable_summaries(b_fc4)
        
        y_conv = tf.matmul(h_fc1_drop, W_fc4) + b_fc4
        tf.summary.histogram('y_conv', y_conv)
        
        
    #Train and Evaluate the Model
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))    
    tf.summary.scalar('training accuracy', accuracy)
    
    
    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()
    train_writer=tf.summary.FileWriter('2conv_train', sess.graph)
    test_writer=tf.summary.FileWriter('2conv_test', sess.graph)
    
    predict = tf.argmax(y_conv, 1)
    sess.run(tf.global_variables_initializer())
    
    Y_train = dense_to_one_hot(Y_train, 10)
    Y_val = dense_to_one_hot(Y_val, 10) 
    

    for i in range(20000):
        batchX, batchY = next_batch(X_train, Y_train, 100)
        #batchY = tf.one_hot(batchY, 10)
        if i%100 == 0:
            
            train_accuracy = accuracy.eval(feed_dict={
                x:batchX, y_: batchY, keep_prob: 1.0})
            
            print("step %d, training accuracy %g"%(i, train_accuracy))

#         train_step.run(feed_dict={x: batchX, y_: batchY, keep_prob: 1.0})
        summary, _ = sess.run([merged, train_step],feed_dict={x: batchX, y_: batchY, keep_prob: 1.0})
        train_writer.add_summary(summary, i)

        
    #writeout

    summary, acc = sess.run([merged, accuracy], feed_dict={
        x: X_val, y_: Y_val, keep_prob: 1.0})
    test_writer.add_summary(summary, i)

    
    print("Validation accuracy %g"%acc)



    saver = tf.train.Saver()
    saver.save(sess, 'CNN_twoLayer')    
    
    #predict
    y_pred = []
    batchsize=100
    for i in range(0, len(X_test), batchsize):
        X_batch = X_test[i:i+batchsize]
        pred = predict.eval(feed_dict={x: X_batch, keep_prob: 1.0})
        y_pred += list(pred)
    #print y_pred
    
    outputFile = 'pred_CNN3Layer_5x32n3x64n1024_withVisualization'    
    with open(outputFile, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['ImageId','label'])
        for l in range(len(y_pred)):
            ImageID = l
            if y_pred[l] == 0:
                label = 10
            else:
                label = y_pred[l] 
            
            info = ([ImageID, label])
            writer.writerow(info)   
        