#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:23:17 2022

@author: tonu
Contains custom for DF only sklearn transformers.
"""
### IMPORTS ###
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


### PREPROCESSING ###

# DF compatible dtype transformer
class DFDtypeTransformer(BaseEstimator, TransformerMixin):
    """Transforms features dtypes by given dtype dictionary.
    
    Parameters
    ----------
    dtypes_dict : dict {'dtype': ['feature1', 'feature2']}
        Dictionary with dtypes as keys and list of features as values.
    
    Returns
    -------
    DataFrame
        Dtypes transforemed data.
    """
    def __init__(self, dtypes_dict: dict):
        self.dtypes_dict = dtypes_dict
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        for dtype, cols_list in self.dtypes_dict.items():
            try:
                X_[cols_list] = X_[cols_list].astype(dtype)
            except KeyError: 
                # if column not in list, remove it
                cols_list_ = [col for col in cols_list if col in X_.columns]
                if len(cols_list_) == 0: continue
                else:
                    X_[cols_list_] = X_[cols_list].astype(dtype)
        return X_

# --- ENCODING, MAPPING --- #

class DFDummiesEncoder(BaseEstimator, TransformerMixin):
    '''Dummie encoding for nominal categorical data, in a
    one-hot-encode fashion. 
    
    Parameters
    ----------
    columns : list, default None
        List of column name(s) to be encoded. If not specified
        all columns are encoded.
    drop_first : bool, default True
        Whether to get k-1 dummies out of k categorical levels 
        by removing the first level.
        
    Returns
    ------- 
    DataFrame
        Dummy-coded data.'''
    def __init__(self, columns=None, drop_first=True):
        self.columns = columns
        self.drop_first = drop_first
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        if self.columns is None:
            self.columns = X_.columns
        return pd.get_dummies(data=X_, 
                              columns=self.columns, 
                              drop_first=self.drop_first)


# --- SCALING, NORMALIZING, UNSKEWING --- #

class DFPowerTransformer(BaseEstimator, TransformerMixin):
    """PowerTransformer to make data more Gaussian-like.
    
    Parameters
    ----------
    columns: list of column names, default None
        List of column name(s) to be transformed. If None all
        columns will be transformed.
    method : str {'yeo-johnson', 'box-cox'}, default 'yeo-johnson'
        The power transformer method. 
        
        - 'yeo-johnson', works with positive and negative values
        - 'box-cox', only works with strictly positive values
        
    standardize : bool, default True
        Apply zero-mean, unit-variance normalization to the
        transformed output, if True.
        
    Returns
    -------
    DataFrmae
        Transformed data."""
        
    def __init__(self, columns=None, method='yeo-johnson', standardize=True):
        self.columns = columns
        self.method = method
        self.standardize = standardize
        self.pt = PowerTransformer(method=self.method,
                                   standardize=self.standardize)
        
    def fit(self, X, y=None):
        self.pt.fit(X, y)
        return self
    
    def transform(self, X, y=None):
        # if X is series or 1 column DF
        if len(X.shape) == 1 or X.shape[1] == 1:
            cols = X.name if len(X.shape) == 1 else X.self.columns
            return pd.DataFrame(
                data=self.pt.transform(X[cols].values.reshape(-1,1)),
                index=X.index,
                columns=cols)
        else:
            return pd.DataFrame(data=self.pt.transform(X), 
                                index=X.index, 
                                columns=X.columns)