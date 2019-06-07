#! /anaconda/bin/python
'''
Created on Feb 9, 2017

@author: ywkim
'''
import csv
from csv import reader
from sys import argv
#from sklearn import linear_model
import numpy as np
from one_vs_all import one_vs_allLogisticRegressor


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

if __name__ == '__main__':
    trainfile = argv[1]
    testfile = argv[2]
    X_train, Y_train = load_csv(trainfile)
    X_test = load_csv_test(testfile)

    ova_logreg = one_vs_allLogisticRegressor(np.arange(1,11))
    
    # train 
    reg = 1.0

    ova_logreg.train(X_train,Y_train,reg)
    

    # predict on test set
    '''
    y_test_pred = np.zeros(shape = (8, 3254)) #empty array
    for i in range(len(X_test)):
        y = ova_logreg.predict(X_test[i])
        y_test_pred[i] = y
        
    y_test_pred = y_test_pred.reshape(1,26032)
    '''
    y_test_pred = ova_logreg.predict(X_test) #the output is an array of indices
    
    
    outputFile = 'pred_OVA_reg0'    
    with open(outputFile, 'w') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(['ImageId','label'])
        for g in range(len(y_test_pred)):
            ImageID = g
            label = y_test_pred[g]+1
            info = ([ImageID, label])
            writer.writerow(info)    
        
    
    
    
    
    
    
    
    
    
    
