#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Python file for accessing custom made seaborn plots.
"""

from IPython.display import clear_output
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def reg_plot(df, x, y):
    """Scatter plot with 1st and optionally 2nd order regression fits
    with 95% confidence intervals.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with all the necessary x and y columns.
    x : str
        X axis column name.
    y : str
        Y axis column name.
    
    Returns
    -------
    None
    """
    
    clear_output() # clear stdout 
    
    #calculate Pearson corr coefs
    r_val1 = pearsonr(df[x], df[y]) 
    r_val2 = pearsonr(df[x]**2, df[y])
    
    #2nd order regression fr only if it improves the 1st order pearson r value
    if abs(r_val2[0]) > abs(r_val1[0]): 
        plt.figure(figsize=(8, 5))
        
        sns.regplot(x=x, y=y, data=df, 
                    line_kws={'color':'red','label':'order 1'})
        sns.regplot(x=x, y=y, data=df, order=2, label='order 2', 
                    color='green', scatter=None)
        
        plt.title('{} r_1={:.2f}, r_2={:.2f}'.format(x, r_val1[0], 
            r_val2[0]), fontsize=14, weight='bold')
        plt.xlabel(x, fontsize=14)
        plt.ylabel(y, fontsize=14)
        
        plt.legend()
        plt.show()
    else:
        plt.figure(figsize=(8, 5))
        
        sns.regplot(x=x, y=y, data=df, 
                    line_kws={'color':'red','label':'order 1'})
        
        plt.title('{} r_1={:.2f}'.format(x, r_val1[0]), 
                  fontsize=14, weight='bold')
        plt.xlabel(x, fontsize=14)
        plt.ylabel(y, fontsize=14)
        
        plt.legend()
        plt.show()

    
def box(df, x, y):
    """Seaborn box plot from DF.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with all the necessary x and y columns.
    x : str
        X axis column name.
    y : str
        Y axis column name.
    
    Returns
    -------
    None
    """
    #order classes by median valus and plot them descendingly
    my_order = df.groupby(by=x)[y].mean().sort_values(ascending=False).index
    
    plt.figure(figsize=(9, 5))
    sns.boxplot(x=x, y=y, data=df, order=my_order)
    
    plt.title('Training data swarms', fontsize=16, weight='bold')
    plt.xlabel(x, fontsize=16)
    plt.ylabel(y, fontsize=16)
    
    plt.show()