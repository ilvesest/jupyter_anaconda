#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:05:11 2020

@author: tonu_ilves

Custom made sklearn Transformers and Estimators
"""

from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, clone
from sklearn.preprocessing import PowerTransformer
from sklearn.compose import ColumnTransformer

from scipy.special import boxcox1p
import numpy as np
import pandas as pd


### DF TRANSFORMERS ###

class DFImputer(BaseEstimator, TransformerMixin):
    """Imputes DF missing values based on fill_value.
    
    Parameters
    ----------
    cols : list of str
        List of DF column names to be imputed. If None, all the columns are imputed.
        this is useful if the transformer is used in a ColumnTransformer pipeline. If
        cols are specified, only these columns in the DF are transformed and others are
        passed through.
    fill_value : str / int / float
        Strategy or value to be filled. Available strategies are ['mean','median','most_frequent'].
        If anything else is passed, the argument is interpreted to be a constant.
        
    Returns
    -------
    df[cols] : DataFrame
        Returns DF with imputed columns if cols=None.
    df : DataFrame
        Returns full df (imputed and not imputed features), if cols are specified.
    """
    def __init__(self, cols=None, fill_value='mean'):
        self.fill_value = fill_value
        self.cols = cols
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if self.cols == None:
            if self.fill_value == 'mean':
                return X.fillna(X.mean())
            elif self.fill_value == 'median':
                return X.fillna(X.median())
            elif self.fill_value == 'most_frequent':
                return X.fillna(X.mode().iloc[0])
            else:
                return X.fillna(self.fill_value)
        else:
            if self.fill_value == 'mean':
                X[self.cols] = X[self.cols].fillna(X[self.cols].mean())
            elif self.fill_value == 'median':
                X[self.cols] = X[self.cols].fillna(X[self.cols].median())
            elif self.fill_value == 'most_frequent':
                X[self.cols] = X[self.cols].fillna(X[self.cols].mode().iloc[0])
            else:
                X[self.cols] = X[self.cols].fillna(self.fill_value)
            return X

class DFPowerTransformer(BaseEstimator, TransformerMixin):
    """PowerTransformer that returns trasnformed data as DF."""
    def __init__(self):
        self.pt = PowerTransformer()
    def fit(self, X, y=None):
        self.pt.fit(X, y)
        return self
    def transform(self, X, y=None):
        return pd.DataFrame(self.pt.transform(X), X.index, X.columns)

class PassThrough(BaseEstimator, TransformerMixin):
    """Dummy transformer for DFColumnTransformer for capturing feature names."""
    def __init__(self):
        pass
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        return X

class DFColumnTransformer(BaseEstimator, TransformerMixin):
    """ColumnTransformer that returns a DF.
    
    Parameters
    ----------
    transformers : list of tuples
        List of tuples in tatures that were not selected. Possible values are
        ['drop', 'passthrough'].
    
    Returns: DataFrame
        DF of transformed data.
    """
    def __init__(self, transformers, remainder='drop'):
        self.transformers = transformers
        self.ct = ColumnTransformer(self.transformers)
        self.remainder = remainder
    def fit(self, X, y=None): 
        if self.remainder == 'drop':
            self.ct.fit(X, y)
            
        elif self.remainder == 'passthrough':
            transformed_features = np.concatenate([tpl[-1] for tpl in self.transformers]) 
            passed_features = list(set(X.columns).difference(transformed_features))
            self.transformers.append(('passthrough', PassThrough(), passed_features))
            self.ct.set_params(transformers=self.transformers).fit(X, y)
            
        return self
    def transform(self, X, y=None):
        col_names = np.concatenate([tple[-1] for tple in self.ct.transformers])
        return pd.DataFrame(data=self.ct.transform(X),
                            index=X.index, columns=col_names)

class DFOrdinalsEncoder(BaseEstimator, TransformerMixin):
    """Encodes categorical features as ordinal based on mapping.
        
        Parameters
        ----------
        mapping_dicts : list of dicts
            List of dictionaries in a form of [{'cols':['col1','coln'],'map':{'None':0,a:1,b:2}}].
            'cols' key points to a list of column names whcih mappings, original to encoded label, 
            are the same."""
    def __init__(self, mapping_dicts):
        self.mapping_dicts = mapping_dicts
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        X_ = X.copy()
        for col in X_.columns:
            for dict_ in self.mapping_dicts:
                if col in dict_['cols']:
                    X_[col] = X_[col].map(dict_['map'])
                    break
        return X_

class DFValueRemapper(BaseEstimator, TransformerMixin):
    def __init__(self, remap_dict={}, vsrest_list=[]):
        self.remap_dict = remap_dict
        self.vsrest_list = vsrest_list
    def fit(self, X, y=None):
        return self
    def transform(self, X, y=None):
        if len(self.vsrest_list) == 0:
            return X[X.columns].replace(to_replace=self.remap_dict)
        else:
            for col in X.columns:
                X[col] = X[col].apply(lambda x: x if x in self.vsrest_list else 'REST')
            return X

