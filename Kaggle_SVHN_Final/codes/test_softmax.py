#! /anaconda/bin/python
'''
Created on Mar 18, 2017

@author: ywkim
'''
import csv
from csv import reader
from sys import argv
#from sklearn import linear_model
import numpy as np
from softmax import softmax_loss_vectorized
import linear_classifier
from IPython import embed

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

def preprocess(X_train,X_val, X_test):

    # Preprocessing: reshape the image data into rows
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
    # second: subtract the mean image from train and test data
    
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    
    # third: append the bias dimension of ones (i.e. bias trick) so that our softmax regressor
    # only has to worry about optimizing a single weight matrix theta.
    # Also, lets transform data matrices so that each image is a row.
    
    X_train = np.hstack([np.ones((X_train.shape[0], 1)), X_train])
    X_val = np.hstack([np.ones((X_val.shape[0], 1)), X_val])
    X_test = np.hstack([np.ones((X_test.shape[0], 1)), X_test])
    
    print 'Training data shape with bias term: ', X_train.shape
    print 'Validation data shape with bias term: ', X_val.shape
    print 'Test data shape with bias term: ', X_test.shape
    
    return X_train, X_val, X_test


def weightedAverage(pixel):
    return 0.299*pixel[0] + 0.589*pixel[1] + 0.114*pixel[2]  

if __name__ == '__main__':
    trainfile = argv[1]
    testfile = argv[2]
    X_train, Y_train = load_csv(trainfile)
    X_test = load_csv_test(testfile)
    
    '''
    #convert rgb->grayscale
    for i in range(73257):
        image = X_train[i]
        gray = np.zeros((image.shape[0], image.shape[1]))
        for rownum in range(len(image)):
            for colnum in range(len(image[rownum])):
                gray[rownum][colnum] = weightedAverage(image[rownum][colnum])

        X_train[i] = np.array([gray]*3).reshape(32,32,3)
    '''
    
    X_train, Y_train, X_val, Y_val = subsample(66257, 7000, X_train, Y_train)
    X_train, X_val, X_test = preprocess(X_train, X_val, X_test)

    theta = np.random.randn(3073, 10)*0.0001
    #loss_vectorized, grad_vectorized = softmax_loss_vectorized(theta, X_train, Y_train, 0.00001)
    
    '''
    results = {}
    best_softmax = linear_classifier.Softmax()
    learning_rates = [1e-7, 5e-7, 1e-6, 5e-6]
    regularization_strengths = [5e4, 1e5, 5e5, 1e8]
    
    for lr in learning_rates:
        for reg in regularization_strengths:
            best_softmax.train(X_train, Y_train, learning_rate=lr, reg=reg, batch_size=600, num_iters=4000)
            #y_test_pred = best_softmax.predict(X_test)
            y_val_pred = best_softmax.predict(X_val)
            y_val_accuracy = np.mean(y_val_pred==Y_val)
            results[(lr, reg)] = (y_val_accuracy)
            
    # Print out results.
    for lr, reg in sorted(results):
        val_accuracy = results[(lr, reg)]
        print 'lr %e reg %e val accuracy: %f' % (
                    lr, reg, val_accuracy)

    '''
    best_softmax = linear_classifier.Softmax()
    best_softmax.train(X_train, Y_train, learning_rate=5.000000e-07, reg=1.000000e+05, batch_size=600, num_iters=4000)
    y_test_pred = best_softmax.predict(X_test)
    '''
    '''
    outputFile = 'pred_SOFTMAX'    
    with open(outputFile, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['ImageId','label'])
        for g in range(len(y_test_pred)):
            ImageID = g
            label = y_test_pred[g]+1
            info = ([ImageID, label])
            writer.writerow(info)    

    
    
    
    
    
    
    
    
    
