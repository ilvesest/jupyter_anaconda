# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a file where I (Tonu) have stored some of my own functions. 
"""

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