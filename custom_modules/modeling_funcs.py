#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 11:58:12 2022

@author: tonu
"""

### ---- IMPORTS ---- ###
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error


### ---- PRE-MODELING ---- ###

def split_data(X, y, test_size=0.25, shuffle=False, **kwargs) -> list:
    """sklearn sklearn.model_selection.train_test_split wrapper."""
    return train_test_split(X, y, test_size=test_size,shuffle=shuffle,**kwargs)

# function to return fit and prediction
def return_fit_test(X_train, X_test, y, model, clip=False):
    """Return model y_fit and y_pred as DF."""
    
    y_fit = pd.DataFrame(model.predict(X_train), 
                         index=X_train.index,
                         columns=y.columns)
    y_pred = pd.DataFrame(model.predict(X_test),
                          index=X_test.index,
                          columns=y.columns)
    if clip:
        return y_fit.clip(.0), y_pred.clip(.0)
    else:
        return y_fit, y_pred
    
### ---- POST-MODELING ---- ###

def rmsle_scores(y_train, y_test, y_fit, y_pred):
    """Wrapper to mean_squared_log_error."""
    
    rmsle_train = mean_squared_log_error(y_train, y_fit) ** 0.5
    rmsle_valid = mean_squared_log_error(y_test, y_pred) ** 0.5
    print(f'Training RMSLE: {rmsle_train:.5f}')
    print(f'Validation RMSLE: {rmsle_valid:.5f}')
    
def rmsle(y_true, y_pred):
    """Square root of MSLE."""
    return mean_squared_log_error(y_true, y_pred) ** 0.5