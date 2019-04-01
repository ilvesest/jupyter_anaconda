#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:53:51 2019

@author: tonu_ilves
"""
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

"""
Recepy for building sklearn-based transformers:
    1. Need to build a class containing 3 methods:
        * fit()
        * transform()
        * fit_transform()
    2. fit() and transform() templete can be obtained from BaseEstimator base 
       class. get_params() and set_params() are also included with BaseEstimator 
    3. fit_transform() can be obtained from TransformerMixin from sklearn
"""

class FeatureSelector(BaseEstimator, TransformerMixin):
    """
    Select specified features from a DF. 
    Parameters:
        feature_names [list] - list of feature names
    """
    def __init__(self, feature_names_list):
        self.feature_names_list = feature_names_list
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.feature_names_list].values

# class FeatureCombiner(BaseEstimator, TransformerMixin):
#     """
#     Create new features based on old features.
#     Parameters:
#         old_features [list] - list of feature names
#         divide_with_feature [str/int] - feature name or number which is used to divide with
#     """
#     def __init__(self, old_feature_indexes, divide_with_feature_index):
#         self.old_feature_indexes = old_feature_indexes
#         self.divide_with_feature_index = divide_with_feature_index
        
#     def fit(self, X, y=None):
#         return self
    
#     def transform(self, X, y=None):
#         new_features = np.array([X[:, i] / X[:, self.divide_with_feature_index] \
#             for i in self.old_feature_indexes])
#         return np.c_[X, np.transpose(new_features)]

        