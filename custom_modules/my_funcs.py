# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a file where I (Tonu) have stored some of my own functions. 
"""

import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats

### DECORATORS ###

def time_it(function):
    """Timing function execution duration.
    
    Parameters:
        function (callable): function to be decorated.
    
    Returns (str): Elapsed time with appropriate prefix.
    """
    
    from time import time
    import math
    
    def wrapper(*args, **kw):
        before = time()
        return_value = function(*args, **kw)
        after = time()
        
        # converting to appropriate second prefix
        units = ['s', 'ms', 'μs', 'ns', 'ps']
        n = float(after-before)
        index = max(0, min(len(units) - 1, 
                           int(abs(math.floor(0 if n == 0 else math.log10(abs(n)) / 3)))))
        print('\nElapsed time: {:.2f} {}'.format(n * 10 ** (index * 3), units[index]))
        return return_value
    return wrapper

### FUNCTIONS ###

def to_file(filename, data, extension=''):
    """Writes data to a file.
    
    Parameters:
        filename (str): name of the file
        data (str): data to be written on the file
        
    Keyword arguments:
        extension (str): file extension with the dot (default '')
        
    Returns: None
    """

    with open(filename + extension, 'w') as file:
        file.write(data)
        
    print('File "{}{}" created.'.format(filename, extension))

### EDA ###

def eda_feature(df, feature, target, bins=50):
    '''Plots box plots or a hstogram of a feature based on if it is
    categorical or continuous-numeric type.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame of the data.
    feature : str
        Feature column name in the df.
    target : str
        Target coumn name in the df.
    bins : int
        Number of bins to plot if feature is target.
    
    Returns
    -------
    None
    '''
    
    from IPython.display import clear_output
    import bokeh_plots as bp
    
    #clear the output field 
    clear_output()
    
    #based on the number of unique values it has
    #determine if feature is categorical or continuous
    n_uniques = len(df[feature].unique())
    if df[feature].dtype == 'O' or n_uniques <= 20:
        
        #print normalized value counts of the classes
        print(df[feature].value_counts(normalize=True, dropna=False))
        
        # plot box plots of the feature vs target
        bp.box(df=df, x=feature, y=target)
    else:
        
        #print the number of nan values
        print('{} NaN-s: {}/{}'.format(feature, df[feature].isna().sum(),
                                       len(df[feature])))
        
        #if feature is continuous plot a histogram
        bp.hist(df=df, feature=feature, bins=bins)

def eda_features(df, target, bins=50):
    '''Inspecting and grouping features into discrete (input "d"), 
    continuous (input "c") and potentially features 
    to drop (nput "x") lists.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame to be explored
    target : str
        Target column name in the df.
    bins : Number of bins to be shown when target is continuous.
    
    Returns
    -------
    discrete : list
        Categorical feature names.
    continuous : list
        Continuous feature names.
    drop: list
        Potential feature names to be dropped.
    '''
    
    from IPython.display import clear_output
    
    discretes = []
    continuous = []
    to_drop = []
    def wrapper():
        '''Wrapper function for automating feature 
        inspection and categorizaton.'''
        for feature in df:
            
            #if iteration/generation reaches target to break the loop
            if feature == target:
                clear_output()
                break
            
            eda_feature(df, feature, target, bins=bins)
            inp = input()
            if inp == 'd':
                yield discretes.append(feature)
            elif inp == 'c':
                yield continuous.append(feature)
            elif inp == 'x':
                yield to_drop.append(feature)
                break
                
    
    for gen in wrapper():
        gen
    return discretes, continuous, to_drop

def feat_corr(feature, target, df, figsize=(7,5)):
    """Scatter plot with 1st and/or 2nd Order polynomial fit and
    95% confidence interval.
    
    Parameters
    ----------
    feature : str
        Feature name in the DF.
    target : str
        Target name in the DF.
    df : DataFrame
    figsize : tuple of len 2
        Tuple of figure dimensions. Default (7,5).
    
    Retruns
    -------
    None"""
    
    r_value1 = stats.pearsonr(df[feature], df[target]) #1st order Pearson corr coef
    r_value2 = stats.pearsonr(df[feature]**2, df[target]) #2nd order Pearson corr coef
    
    if abs(r_value2[0]) > abs(r_value1[0]): 
    # plot y(x) with regression line and uncertainty area
        plt.figure(figsize=figsize)
        sns.regplot(x=feature, y=target, data=df, line_kws={'color':'red','label':'order 1'})
        sns.regplot(x=feature, y=target, data=df, order=2, label='order 2', color='green', scatter=None)
        plt.title('{} r_1={:.2f}, r_2={:.2f}'.format(feature, r_value1[0], r_value2[0]), fontsize=14, weight='bold')
        plt.xlabel(feature, fontsize=14)
        plt.ylabel(target, fontsize=14)
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=figsize)
        sns.regplot(x=feature, y=target, data=df, line_kws={'color':'red','label':'order 1'})
        plt.title('{} r_1={:.2f}'.format(feature, r_value1[0]), fontsize=14, weight='bold')
        plt.xlabel(feature, fontsize=14)
        plt.ylabel(target, fontsize=14)
        plt.legend()
        plt.show()

def feats_squared_corr(features, target, df):  
    """Scatter plot of feature and target if 2nd Order poly fit R^2 is greater
    than 1st Order poly fit R^2.
    
    Parameters
    ----------
    features : str | list of str
        Feature name or list of feature names.
    target : str
        Target name in the df.
    df : DataFrame
    
    Returns
    -------
    None
    """ 
    
    if type(features) == str:
        r_value1 = stats.pearsonr(df[features], df[target]) #1st order Pearson corr coef
        r_value2 = stats.pearsonr(df[features]**2, df[target]) #2nd order Pearson corr coef
        
        if abs(r_value2[0]) > abs(r_value1[0]):
            plt.figure(figsize=(7,5))
            sns.regplot(x=features, y=target, data=df, line_kws={'color':'red','label':'1st Order'})
            sns.regplot(x=features, y=target, data=df, order=2, label='2nd Order', color='green', scatter=None)
            plt.title('{} r_1={:.2f}, r_2={:.2f}'.format(features, r_value1[0], r_value2[0]), 
                      fontsize=14, weight='bold')
            plt.xlabel(features, fontsize=14)
            plt.ylabel(target, fontsize=14)
            plt.legend()
            plt.show()
        else:
            print('2nd Order correlation did not exceed 1st Order.')
    else:
        
        squares = []
        for f in features:
            r1 = stats.pearsonr(df[f], df[target])
            r2 = stats.pearsonr(df[f]**2, df[target])
            if abs(r2[0]) > abs(r1[0]):
                squares.append((f,r1[0],r2[0]))

        if len(squares) == 0:
            return '2nd Order correlation did not exceed 1st Order for all features.'
        
        elif len(squares) == 1:
            plt.figure(figsize=(7,5))
            sns.regplot(x=features, y=target, data=df, line_kws={'color':'red','label':'1st Order'})
            sns.regplot(x=features, y=target, data=df, order=2, label='2nd Order', color='green', scatter=None)
            plt.title('{} r_1={:.2f}, r_2={:.2f}'.format(features, r_value1[0], r_value2[0]), 
                      fontsize=14, weight='bold')
            plt.xlabel(features, fontsize=14)
            plt.ylabel(target, fontsize=14)
            plt.legend()
        
        else:
            rows = math.ceil(len(squares) / 3)
            fig, axes = plt.subplots(rows,3, figsize=(20,20), 
                                     subplot_kw={'xticks':(), 'yticks':()})

            for i, (ax,tpl) in enumerate(zip(axes.ravel(), squares)):
                if i >= len(squares):
                    break
                else:
                    ax.set_title('{} r_1={:.2f}, r_2={:.2f}'.format(tpl[0], tpl[1], tpl[2]), 
                                 fontsize=12, weight='bold')
                    ax.set_xlabel(tpl[0], fontsize=14)
                    ax.set_ylabel(target, fontsize=14)
                    sns.regplot(x=tpl[0], y=target, data=df, line_kws={'color':'red','label':'1st Order'}, ax=ax)
                    try:
                        sns.regplot(x=tpl[0], y=target, data=df, order=2, label='2nd Order', 
                                    color='green', scatter=None, ax=ax)
                    except ValueError:
                        pass
                    ax.legend()
            for ax in axes.ravel()[len(squares):]:
                ax.set_visible(False)
        plt.show()


### ### ###

def si_prefix(number, unit):
    """Convert a value into specified (SI system) prefix-unit representation.
    
    Parameters:
        number (int/float): number to be converted 
        unit (str): physical unit to be used
        
    Returns: (str)
    """
    
    import math
    
    n = float(abs(number))
    
    if abs(number) > 0 and abs(number) < 1:
        neg_symbols = ['','m','μ','n','p']
        
        #finding appropriate negative prefix index
        i = max(0, min(len(neg_symbols)-1, 
                   int(abs(math.floor(math.log10(n) / 3)))))
        
        return '{:.2f} {}'.format(number * 10 ** (i * 3), neg_symbols[i]+unit)
    
    else:
        pos_symbols = ['','k','M','G','T']
        
        #finding appropriate positive prefix index
        i = max(0, min(len(pos_symbols)-1, 
                   int(math.floor(math.log10(abs(n)) / 3))))
        
        return '{:.2f} {}'.format(number / 10 ** (i * 3), pos_symbols[i]+unit)
    
### ### ###
        
def all_errors():
    """Returns all possible Python errors.
    
    Returns: (list)
    """
    
    import re
    
    return sorted([x for x in __builtins__.__dict__.keys() if re.compile(r"^.*Error").search(x)])