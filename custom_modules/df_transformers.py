#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 14 09:23:17 2022

@author: tonu
Contains custom for DF only sklearn transformers.
"""
### IMPORTS ###
import pandas as pd
import numpy as np


from sklearn.base import BaseEstimator, TransformerMixin, RegressorMixin, \
    clone
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

"""
class SKELETON(BaseEstimator, TransformerMixin):
    def __init__(self):
    def fit(self, X, y=None):
        self.all_columns_ = X.columns
        return self
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return self.all_columns_
    def transform(self, X, y=None):
        return X
"""


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

class DFDummyEncoder(BaseEstimator, TransformerMixin):
    """Convert categorical variable into dummy/indicator variables.
    Built on top of pandas get_dummies() function.
    
    Parameters
    ----------
    drop_first : bool, default False
        Whether to get k-1 dummies out of k categorical levels by removing 
        the first level.
    
    Returns
    -------
    DataFrame
        Dummy-coded data.
    """
    def __init__(self, drop_first=False):
        self.drop_first = drop_first
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        X_ = X.copy()
        return pd.get_dummies(X_, drop_first=self.drop_first)

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
        self.with_mean = with_mean
        self.with_std = with_std
        self.sc = StandardScaler(with_mean=self.with_mean, with_std=self.with_std)
        
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

# --- MAPPING --- #
class DFDtypeMapper(BaseEstimator, TransformerMixin):
    """Remap pandas dataframe dtypes.
    Parameters
    ----------
    dtype_dict : dict, {'dtype':[col_name]}
        Dictionary of dtypes as keys and values as list of column names. 
    
    Returns
    -------
    DataFrame : pd.DataFrame"""
    def __init__(self, dtype_dict : dict):
        self.dtype_dict = dtype_dict
        self.transformed_column_names = None 
    
    def fit(self, X, y=None):
        self.all_columns_ = X.columns
        return self
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return self.all_columns_
    
    def transform(self, X, y=None) -> pd.DataFrame:
        X_ = X.copy()
        # remove columns that are not in X
        _dtype_dict = {}
        for dtype, val in self.dtype_dict.items():
            if isinstance(val, str):
                if val in X_.columns: 
                    _dtype_dict[dtype] = val
            elif type(val) not in [tuple, list, np.ndarray]:
                raise ValueError(f'Wrong type for {self.dtype_dict} value.')
            else:
                _dtype_dict[dtype] = [col for col in val 
                                           if col in X_.columns]
        
        for dtype in _dtype_dict:
            X_[_dtype_dict[dtype]] = X_[_dtype_dict[dtype]].astype(dtype)
        
        return X_

class DFValueMapper(BaseEstimator, TransformerMixin):
    """Rename values in column based on dictionary.
    Parameters
    ----------
    map_dict : dict 
        Dictionary of old mappings to new.
    cat_only : bool, default True
        - If True: consider category dtype columns only
        - If False: apply to all columns. Computationally more expensve.
    
    Returns
    -------
    DataFrame : pd.DataFrame
        Remapped pandas DataFrame."""
    def __init__(self, map_dict : dict, cat_only=True):
        self.cat_only = cat_only
        self.map_dict = map_dict
    def fit(self, X, y=None):
        self.all_columns_ = X.columns
        return self
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        check_is_fitted(self)
        return self.all_columns_
    def transform(self, X, y=None) -> pd.DataFrame:
        X_ = X.copy()
        # categorical features
        if self.cat_only:
            cat_cols = X_.columns[(X_.dtypes == 'category').values]
            X_[cat_cols] = X_[cat_cols].apply(
                lambda x: x.cat.rename_categories(self.map_dict))
            return X_
        else:
            return X_.replace(self.map_dict)

# --- PIPING --- #

class DFColumnTransformer(BaseEstimator, TransformerMixin):
    """Applies transformers to columns of an array or pandas DataFrame.
    
    Parameters
    ----------
    transformers : list of tuples
        Tuples of the form: (name, transformer, columns).
        name : str
            Step name, allows he transformer and its parameters to be set 
            using set_params and searched in grid search.
        transformer : {‘drop’, ‘passthrough’} or estimator
            Estimator must support fit and transform. Special-cased strings 
            ‘drop’ and ‘passthrough’ are accepted as well, to indicate to 
            drop the columns or to pass them through untransformed, 
            respectively.
        columns : str, array-like of str, etc
            Strings can reference DataFrame columns by name.
    remainder : {‘drop’, ‘passthrough’}, default='drop'
        Strategy for the features that were not selected.
    n_jobs : int, default=-1
        None means 1 and -1 means all available processors.
    verbose_feature_names_out : bool, default=False
        If True, get_feature_names_out will prefix all feature names with 
        the name of the transformer that generated that feature.
    
    Returns
    -------
    DataFrame
        DF of transformed data.
    """
    def __init__(self, transformers, remainder='drop', n_jobs=-1,
                 verbose_feature_names_out=False):
        self.transformers = transformers
        self.remainder = remainder
        self.n_jobs = n_jobs
        self.verbose_feature_names_out=verbose_feature_names_out
        self.ct = ColumnTransformer(
            transformers, 
            remainder=remainder,
            n_jobs=n_jobs,
            verbose_feature_names_out=verbose_feature_names_out)
        
        # init transformed column names
        self.transformed_column_names: list[str] = None
        
    def _get_column_names(self, X: pd.DataFrame) -> list[str]:
        """Get names of transformed columns from a fitted self.ct
        
        Parameters
        ----------
        X : DataFrame
            DataFrame to be fitted on
        
        Returns
        -------
        column_names : List[str]
            Flattened list of column names.
        """
        cols_lists = []
        for name, transformer, cols in self.ct.transformers_:
            
            if hasattr(transformer, "get_feature_names_out"):
                cols_lists.append(transformer.get_feature_names_out(cols))
            # select by remainder
            elif name=='remainder' and self.ct.remainder=="passthrough":
                cols_lists.append(X.columns[cols]) # cols are ints
            # select by transformer
            elif isinstance(transformer,str) and transformer=='passthrough':
                cols_lists.append(cols)
            # drop by remainder or transformer
            elif (name == "remainder" and self.ct.remainder == 'drop') or \
                (isinstance(transformer,str) and transformer == 'drop'):
                continue
            else:
                cols_lists.append(cols)
        
        # flatten the list
        return [col for list_ in cols_lists for col in list_]
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformation
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.
        
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if self.transformed_column_names is not None:
            return self.transformed_column_names
        else:
            raise ValueError(f"{self} is not fitted!")
    
    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit ColumnTransformer, and obtain names of transformed columns."""
        assert isinstance(X, pd.DataFrame)
        
        self.ct.fit(X, y)
        self.transformed_column_names = self._get_column_names(X)
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform a new DF using fitted self.ct."""
        assert isinstance(X, pd.DataFrame)
        
        transformed_X = self.ct.transform(X)
        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(
                data=transformed_X,
                index=X.index,
                columns=self.transformed_column_names)
        else:
            return pd.DataFrame.sparse.from_spmatrix(
                data=transformed_X,
                index=X.index,
                columns=self.transformed_column_names)

class DFPipeline(BaseEstimator, TransformerMixin):
    """A wrapper to sklearn.pipeline.Pipeline to return
    data as pandas DataFrame with corresponding column names."""
    
    def __init__(self, steps, **kwargs):
        """Initialize Pipeline object through Pipeline construct.
        
        Parameters
        ----------
        steps : list of tuples
            List of (name, transform) tuples (implementing fit/transform) 
            that are chained.
        kwargs : keyword arguments for sklearn.pipeline.Pipeline.
        """
        self.steps = steps
        self.pipeline = Pipeline(steps, **kwargs)
        self.transformed_column_names = None
        
    def _get_column_names(self, X: pd.DataFrame) -> list:
        """Get names of transformed columns from a fitted self.pipeline
        
        Parameters
        ----------
        X : DataFrame
            DataFrame to be fitted on
        
        Returns
        -------
        column_names : List[str]
            Flattened list of column names.
        """
        # last step in the pipeline
        last_estimator = self.pipeline.get_params()['steps'][-1][-1]
        input_features = X.columns
        
        if hasattr(last_estimator, "get_feature_names_out"):
            return last_estimator.get_feature_names_out(input_features)
        elif hasattr(last_estimator, "feature_names_in_") and \
            last_estimator.feature_names_in_ is not None:
            return list(last_transformer.feature_names_in_)
        else:
            return input_features
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformation
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.
        
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if self.transformed_column_names is not None:
            return self.transformed_column_names
        else:
            raise ValueError(f"{self} is not fitted!")
    
    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit Pipeline, and obtain names of transformed columns."""
        assert isinstance(X, pd.DataFrame)
        self.pipeline.fit(X, y)
        
        self.transformed_column_names = self._get_column_names(X)
        return self
    
    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform a new DF using fitted self.pipeline."""
        assert isinstance(X, pd.DataFrame)
        
        transformed_X = self.pipeline.transform(X)
        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(
                data=transformed_X,
                index=X.index,
                columns=self.transformed_column_names)
        else:
            return pd.DataFrame.sparse.from_spmatrix(
                data=transformed_X,
                index=X.index,
                columns=self.transformed_column_names)

# NB! DFmake_ct works with fit and trasnform but can throw RuntimeError:
# RuntimeError: scikit-learn estimators should always specify their parameters 
# in the signature of their __init__ (no varargs). <class '__main__.DFmake_ct'> 
# with constructor (self, *transformers, remainder='drop', n_jobs=-1, 
# verbose_feature_names_out=False) doesn't  follow this convention.
class DFmake_ct(BaseEstimator, TransformerMixin):
    """Applies transformers to columns of an array or pandas DataFrame..
    
    Parameters
    ----------
    *transformers : tuples
        Tuples of the form: (transformer, columns).
    remainder : {‘drop’, ‘passthrough’}, default='drop'
        Strategy for the features that were not selected.
    n_jobs : int, default=-1
        None means 1 and -1 means all available processors.
    verbose_feature_names_out : bool, default=False
        If True, get_feature_names_out will prefix all feature names with 
        the name of the transformer that generated that feature.
    
    Returns
    -------
    DataFrame
        DF of transformed data.
    """
    def __init__(self, *transformers, remainder='drop', n_jobs=-1, 
                 verbose_feature_names_out=False):
        self.transformers = transformers
        self.remainder = remainder
        self.n_jobs = n_jobs
        self.transformed_column_names = None
        self.ct = make_column_transformer(
            *transformers,
            remainder=remainder,
            n_jobs=n_jobs,
            verbose_feature_names_out=verbose_feature_names_out)
        
    def _get_column_names(self, X: pd.DataFrame):
        """Get names of transformed columns from a fitted self.ct
        
        Parameters
        ----------
        X : DataFrame
            DataFrame to be fitted on
        
        Returns
        -------
        column_names : List[str]
            Flattened list of column names.
        """
        cols_lists = []
        for name, transformer, cols in self.ct.transformers_:
            
            if hasattr(transformer, "get_feature_names_out"):
                cols_lists.append(transformer.get_feature_names_out(cols))
            # select by remainder
            elif name=='remainder' and self.ct.remainder=="passthrough":
                cols_lists.append(X.columns[cols]) # cols are ints
            # select by transformer
            elif isinstance(transformer,str) and transformer=='passthrough':
                cols_lists.append(cols)
            # drop by remainder or transformer
            elif (name == "remainder" and self.ct.remainder == 'drop') or \
                (isinstance(transformer,str) and transformer == 'drop'):
                continue
            else:
                cols_lists.append(cols)
        
        return [col for list_ in cols_lists for col in list_]
    
    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Get output feature names for transformation
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features.
        
        Returns
        -------
        feature_names_out : ndarray of str objects
            Transformed feature names.
        """
        if self.transformed_column_names is not None:
            return self.transformed_column_names
        else:
            raise ValueError(f"{self} is not fitted!")
    
    def fit(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fit ColumnTransformer, and obtain names of transformed columns."""
        assert isinstance(X, pd.DataFrame)
        
        self.ct.fit(X, y)
        self.transformed_column_names = self._get_column_names(X)
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """Transform a new DF using fitted self.ct."""
        assert isinstance(X, pd.DataFrame)
        
        transformed_X = self.ct.transform(X)
        if isinstance(transformed_X, np.ndarray):
            return pd.DataFrame(
                data=transformed_X,
                index=X.index,
                columns=self.transformed_column_names)
        else:
            return pd.DataFrame.sparse.from_spmatrix(
                data=transformed_X,
                index=X.index,
                columns=self.transformed_column_names)

################### --- ESTIMATORS --- ###################

####### ENSEMBLES #######

class AveragingEnsemble(BaseEstimator, RegressorMixin, TransformerMixin):
    def __init__(self, models):
        self.models = models
    
    # define clones of the base models
    def fit(self, X, y):
        self.models_ = [clone(model) for model in self.models]
        self.y = y
        #train the models
        for model in self.models_:
            model.fit(X, y)
        
        return self
    
    # averaging predictions
    def predict(self, X):
        predictions = [model.predict(X).clip(min=0) for model in self.models_]
        predictions = sum(predictions) / len(predictions) 
        return predictions

class CustomStackingRegressor(BaseEstimator, TransformerMixin):
    
    def __init__(self, model_1, model_2):
        self.model_1  = clone(model_1)
        self.model_2 = clone(model_2)
        self.y_columns = None # dummy for column names from fit method later
        
    def fit(self, X, y):
        self.model_1.fit(X, y) # fit 1st model
        
        # first model prediction that captures trend
        y_pred_1 = pd.DataFrame(self.model_1.predict(X), 
                                index=X.index, 
                                columns=y.columns).clip(0.0)
        
        y_resid = y - y_pred_1
        
        # fit 2nd model on features designed for it 
        self.model_2.fit(X, y_resid)
        
        self.y_columns = y.columns # col names for predict method
        self.y_pred_1 = y_pred_1
        self.y_resid = y_resid
        
        return self
    
    def predict(self, X):
        y_pred = pd.DataFrame(self.model_2.predict(X),
                              index=X.index, 
                              columns=self.y_columns)
        
        return y_pred.clip(0.0)