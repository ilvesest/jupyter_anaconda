#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 10:38:39 2019

@author: tonu_ilves
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

class Knn:
    '''Classification and regression machine learning algorithms for
    labeled data.'''
    def __init__(self):
        pass
    
    def info(self):
        c_methods = [f for f in dir(Knn) if not f.startswith('__')]

        reminder = "Reminder!\n\t+ Data as NumPy array or pandas DF\n\t"\
        "+ CANNOT contain MISSING data\n\t+ Feature & target values same length" 
        print(reminder)
        print("\nClass methods:")
        for i in c_methods:
            print("\t+", i)
    
    def complexity_curve(self, X, y, max_neighbors=20, random_state=None, 
                         stratify=None, test_size=0.25):
        '''Plots the complexity curve of kNN Classifier.
            Parameters:
                X - [ndarray,DF] Feature values
                y - [ndarray,DF] Target values
                max_neighbours - [int]
                stratify - [y] Uniformly selects target values for training
                               and testing data
                test_size [float[0;1]] Fraction of test size
            Returns: None'''
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state,
                                                           stratify=stratify, test_size=test_size)
        neighbors = np.arange(1, max_neighbors+1)
        train_acc = np.empty(len(neighbors))
        test_acc = np.empty(len(neighbors))
        
        for k in neighbors:
            clf = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)
            train_acc[k-1] = clf.score(X_train, y_train)
            test_acc[k-1] = clf.score(X_test, y_test)
        
        sns.set()
        plt.figure(figsize=(10, 8))
        plt.title("Model Complexity Curve", weight='bold', fontsize=18)
        plt.plot(neighbors, train_acc, label='Training Accuracy')
        plt.plot(neighbors, test_acc, label='Test Accuracy')
        plt.legend()
        plt.xlabel("Number of Neighbors", fontsize=14)
        plt.xticks(ticks=neighbors[1::2])
        plt.ylabel("Accuracy", fontsize=14)
        plt.show()
        
    def decision_boundry(self, X, y, n_neighbors, n_targets=2, step=0.2, 
                         figsize=(14, 5), weights=['uniform', 'distance'],
                         weight_names=None):
        '''Plotting decision boundries.'''
        
        h = step  # step size in the mesh
        
        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
        
        plt.figure(figsize=figsize)                            
                                    
        for i, weight in enumerate(weights):
            # we create an instance of Neighbours Classifier and fit the data.
            clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weight)
            clf.fit(X, y)
        
            # Plot the decision boundary. For that, we will assign a color to each
            # point in the mesh [x_min, x_max]x[y_min, y_max].
            x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
            y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                 np.arange(y_min, y_max, h))
            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        
            # Put the result into a color plot
            Z = Z.reshape(xx.shape)
            plt.subplot(1, len(weights), i+1)
            plt.pcolormesh(xx, yy, Z, cmap='gist_ncar')
        
            # Plot also the training points
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis',
                        edgecolor='k', s=20)
            plt.xlim(xx.min(), xx.max())
            plt.ylim(yy.min(), yy.max())
        
            if weight_names is not None:
                plt.title("%i-Class classification (k = %i, weights = '%s')" 
                          % (n_targets, n_neighbors, weight_names[i]))
            else:
                plt.title("%i-Class classification (k = %i, weights = '%s')" 
                          % (n_targets, n_neighbors, weight))
        
        plt.show()