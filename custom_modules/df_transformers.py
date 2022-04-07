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

# --- ENCODING, MAPPING --- #

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
        
        # check if dtypes_dict has proper format
        assert all(type(val) is list for key,val in dtypes_dict.items()), \
            "Dictionary values must be of type list."
        
        self.dtypes_dict = dtypes_dict
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        for dtype, feature_list in self.dtypes_dict.items():
            for feature in feature_list:
                try:
                    X_[feature] = X_[feature].astype(dtype)
                except KeyError: continue
        return X_

class DFImputer(BaseEstimator, TransformerMixin):
    """Fill NA/NaN values using the specified value, strategy or method. 
    Implemented on top of pandas fillna() function.
    
    Parameters
    ----------
    value : scalar, dict, Series, or DataFrame
        Value to use to fill holes (e.g. 0), alternately a
        dict/Series/DataFrame of values specifying which value to use for
        each index (for a Series) or column (for a DataFrame).  Values not
        in the dict/Series/DataFrame will not be filled. This value cannot
        be a list.
    strat : {'mean', 'median', 'most_frequent'}, defualt None
        Aggregation strategy to use for imputing NaN-s. 
    method : {'backfill', 'bfill', 'pad', 'ffill', None}, default None
        Method to use for filling holes in reindexed Series
        pad / ffill: propagate last valid observation forward to next valid
        backfill / bfill: use next valid observation to fill gap.
        
    Returns
    -------
    DataFrame or None
    """
    # asterisk denotes key-word arguments only
    def __init__(self, *, value=None, strat=None, method=None):
        
        # asserting the parameters are set correctly
        if value == strat == method == None:
            raise ValueError("Value, strat and method can't all be None.")
        elif sum(1 for i in (value, strat, method) if i is not None) > 1:
            raise ValueError("Specify one key-word argument only.")
        elif strat is not None:
            if strat not in ['mean', 'median', 'most_frequent']:
                raise ValueError(f"{strat} not recognized as valid parameter.")
        
        self.value = value
        self.strat = strat
        self.method = method
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        
        if self.value is not None: 
            return X_.fillna(self.value)
        
        elif self.strat is not None:
            if self.strat == 'mean':
                return X_.fillna(X_.mean())
            elif self.strat == 'median':
                return X_.fillna(X_.median())
            elif self.strat == 'most_frequent':
                return X_.fillna(X_.mode().iloc[0])
        
        elif self.method is not None:
            return X_.fillna(method=self.method)

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
        X_ = X.copy()
        
        # if X is series or 1 column DF
        if len(X_.shape) == 1 or X_.shape[1] == 1:
            cols = X_.name if len(X_.shape) == 1 else X_.self.columns
            
            return pd.DataFrame(
                data=self.pt.transform(X_[cols].values.reshape(-1,1)),
                index=X_.index,
                columns=cols)
        else:
            # use all columns if columns are not specified
            cols = X_.columns if self.columns is None else self.columns
            
            return pd.DataFrame(data=self.pt.transform(X_[cols]), 
                                index=X_.index, 
                                columns=X_.columns)

class DFStandardScaler(BaseEstimator, TransformerMixin):
    """Standardize features by removing the mean and scaling to 
    unit variance.
    
    Parameters
    ----------
    columns : list of column names, default None
        List of column name(s) to be standardized. If None all
        columns will be transformed.
    with_mean : bool, default=True
        If True, center the data before scaling. This does not work (and will 
        raise an exception) when attempted on sparse matrices, because 
        centering them entails building a dense matrix which in common use 
        cases is likely to be too large to fit in memory.
    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently, unit 
        standard deviation).
    Returns
    -------
    DataFrmae
        Standardized data."""
        
    def __init__(self, columns=None, with_mean=True, with_std=True):
        self.columns = columns
        self.sc = StandardScaler(with_mean=with_mean, with_std=with_std)
        
    def fit(self, X, y=None):
        cols = X.columns if self.columns is None else self.columns
        self.sc.fit(X[cols], y)
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        
        # if X is series or 1 column DF
        if len(X_.shape) == 1 or X_.shape[1] == 1:
            cols = X_.name if len(X_.shape) == 1 else X_.self.columns

            return pd.DataFrame(
                data=self.sc.transform(X_[cols].values.reshape(-1,1)),
                index=X_.index,
                columns=cols)
        else:
            # use all columns if columns are not specified
            cols = X_.columns if self.columns is None else self.columns

            return pd.DataFrame(data=self.sc.transform(X_[cols]), 
                                index=X_.index, 
                                columns=cols)