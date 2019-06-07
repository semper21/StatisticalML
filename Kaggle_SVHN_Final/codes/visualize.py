#! /anaconda/bin/python
'''
Created on Feb 7, 2017

@author: ywkim

1) load data
2) parse data -> separate label 
3) reshape
4) visualize
'''
from csv import reader
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
            label = row[0]
            data = row[1:]
            dataset.append(data)
            labelset.append(label)
    X = np.array(dataset)
    print X.shape
    X = X.reshape(32, 32, 3, -1).transpose(3,0,1,2)
    Y = np.array(labelset)
    return X, Y

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

def visualize(X):
    # Visualize some examples from the dataset.
    # We show a few examples of training images from each class.

    plt.figure(figsize=(20,2))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(X[i].astype('uint8'))
    plt.savefig('train_first10.png')
    #plt.show()



if __name__ == '__main__':
    filename = argv[1]
    #X, Y = load_csv(filename)
    X = load_csv_test(filename)
    #np.savetxt("transposedX.csv", X, delimiter=",")
    visualize(X)
    #print Y[:10]
