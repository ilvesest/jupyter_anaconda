# -*- coding: utf-8 -*-
"""
Spyder Editor

Custom functions in Machine Learning project work flow. 
"""
################### --- IMPORTS --- ###################
import warnings

import pandas as pd
import numpy as np
import math
import os, re

from scipy import stats
from scipy.signal import periodogram

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

from sklearn.metrics import mean_squared_log_error, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV

################### ---  --- ###################
################### --- IO --- ###################


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

def preds_to_csv(y_pred, id_series=test_id) -> None:
    """Align predictions with correct ID and write to csv."""
    # date index to store_nbr - family - date
    y_pred_stacked = (
        y_pred.copy()
        .stack(['store_nbr', 'family'])
        .reset_index()
        .set_index(['store_nbr', 'family', 'date'])
        .sort_index()
        .rename({0:'sales'}, axis='columns'))
    
    # merge preds to ID-s
    y_pred_id = y_pred_stacked.merge(test_id, how='left', 
                left_index=True, right_index=True)
    
    assert y_pred_id.isna().sum().sum() == 0, 'NaN value(s) contained!'
    
    # 'id' as index and 'sales' as values
    y_pred_id = (
    y_pred_id
    .reset_index()
    .set_index('id')
    .sort_index()
    .sales
    .to_frame())
    
    # write df to csv
    relative_path = 'submissions/favorita_store_sales'
    sub_filenames = (os.listdir(os.getcwd() + '/' + relative_path))
    file_n = None
    if len(sub_filenames) == 0:
        file_n = 1
    else:
        file_n = max([int(re.search(r"^sub(\d+).csv$", filename)[1]) \
                      for filename in sub_filenames]) + 1
    
    y_pred_id.to_csv(f"{relative_path}/sub{file_n}.csv", index=True)
    print(f"sub{file_n}.csv created.")
    print(y_pred_id.head(2))

################### --- EDA --- ###################

####### PLOTTING #######
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

### SCATTER ###
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

### TIME-SERIES ###

# PLOTTING #
def seasonal_plot(df, y, period, freq, ax=None):
    """Return seasonal plot axis."""
    
    if ax is None:
        _, ax = plt.subplots()
        
    palette = sns.color_palette("husl", n_colors=df[period].nunique(),)
    ax = sns.lineplot(
        x=freq,
        y=y,
        hue=period,
        data=df,
        ci=False,        # confidence interval
        ax=ax,
        palette=palette,
        legend=False,)
    
    ax.set_title(f"Seasonal Plot ({period}/{freq})")
    
    # annotate period on the plot
    for line, name in zip(ax.lines, df[period].unique()):
        y_ = line.get_ydata()[-1]              # grab last y value for period
        ax.annotate(
            name,
            xy=(1, y_),
            xytext=(6, 0),
            color=line.get_color(),
            xycoords=ax.get_yaxis_transform(),
            textcoords="offset points",
            size=14,
            va="center")
    return ax

def plot_periodogram(ts, detrend='linear', ax=None, color='purple', alpha=1):
    """Plot periodogram."""
    
    fs = pd.Timedelta("365D") / pd.Timedelta("1D")
    freqencies, spectrum = periodogram(
        ts,                 # time series
        fs=fs,
        detrend=detrend,
        window="boxcar",
        scaling='spectrum')
    
    if ax is None:
        _, ax = plt.subplots()
        
    ax.step(freqencies, spectrum, color=color, alpha=alpha)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 4, 6, 12, 26, 52, 104])
    ax.set_xticklabels([
            "Annual (1)",
            "Semiannual (2)",
            "Quarterly (4)",
            "Bimonthly (6)",
            "Monthly (12)",
            "Biweekly (26)",
            "Weekly (52)",
            "Semiweekly (104)",
    ],rotation=30)
    
    _ = ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    _ = ax.set_ylabel("Variance")
    _ = ax.set_title("Periodogram")
    return ax

def lagplot(x, y=None, lag=1, lead=None, standardize=False, ax=None, **kwargs):
    """Plot a lagplot."""
    
    if lead is not None:
        x_ = x.shift(-lead)
    else:
        x_ = x.shift(lag)
        
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    
    corr = y_.corr(x_)
    if ax is None:
        fig, ax = plt.subplots()
        
    scatter_kws = dict(s=3)
    line_kws = dict(color='red')
    
    ax = sns.regplot(x=x_,
                     y=y_,
                     scatter_kws=scatter_kws,
                     line_kws=line_kws,
                     lowess=True,
                     ax=ax,
                     **kwargs)
    at = AnchoredText(
        f"{corr:.2f}",
        prop=dict(size="large", color='grey', 
                  backgroundcolor=ax.get_facecolor()),
        frameon=True,
        loc="upper left",
    )
    at.patch.set_boxstyle("square, pad=0.0")
    ax.add_artist(at)
    if lead is not None:
        ax.set(title=f"Lead {lead}", xlabel=x_.name, ylabel=y_.name)
    else:
        ax.set(title=f"Lag {lag}", xlabel=x_.name, ylabel=y_.name)
    return ax

def plot_lags(x, y=None, lags=0, leads=0, nrows=1, lagplot_kwargs={}, **kwargs):
    """Plot multiple lag plots."""
    
    leadlag_sum = lags + leads
    
    kwargs.setdefault('nrows', nrows)
    kwargs.setdefault('ncols', math.ceil(leadlag_sum / nrows))
    kwargs.setdefault('figsize', (kwargs['ncols'] * 2, nrows * 2 + 0.5))
    
    n_axes = range(kwargs['nrows'] * kwargs['ncols'])
    leadlag_ndx = [*range(leads, 0, -1)] + [*range(1, lags+1)]
    leadlag_name = ['lead'] * leads + ['lag'] * lags
    
    fig, axs = plt.subplots(sharex=True, sharey=True, squeeze=False, **kwargs)
    for ax, k, name, ndx in zip(fig.get_axes(), n_axes, 
                                leadlag_name, leadlag_ndx):

        if k + 1 <= leadlag_sum:
            if name == 'lag':
                ax = lagplot(x, y, lag=ndx, ax=ax, **lagplot_kwargs)
                ax.set_title(f"Lag {ndx}", fontdict=dict(fontsize=14))
                ax.set(xlabel="", ylabel="")
            else:
                ax = lagplot(x, y, lead=ndx, ax=ax, **lagplot_kwargs)
                ax.set_title(f"Lead {ndx}", fontdict=dict(fontsize=14))
                ax.set(xlabel="", ylabel="")
        else:
            ax.axis('off')
        
    fig.supxlabel(x.name, weight='bold')
    fig.supylabel(y.name if y is not None else x.name, weight='bold')
    fig.tight_layout(w_pad=0.1, h_pad=0.1)
    return fig

# ANALYSIS #
def datetime_gaps(df : pd.DataFrame, column : str, freq='D'):
    """Display time series frequencies and gaps.
    
    Parameters
    ----------
    column : str, DataFrame column or index name.
    freq : str, default 'D'
        Predominant frequency of the datetime column/index."""
    
    df = df.reset_index()
    date_range = pd.date_range(df[column][0], df[column].iloc[-1], freq=freq)
    df[column] = df[column].astype(f"period[{freq}]")
    
    # find frequencies
    temp = df.groupby([column]).sum().reset_index()
    freqs = (temp.loc[:,column]# frequencies
        .diff()
        .value_counts(dropna=False)
        .to_frame())
    print("Frequencies")
    print(freqs)
    
    # find gaps
    gaps = date_range.difference(df[column])
    if len(gaps) == 0:
        print(f"No gaps in {column}.")
    else:
        print(f"{len(gaps)} gaps in datetime:")
        return gaps
################### --- DATA SPLITTING --- ###################

def split_data(X, y, test_size=0.25, shuffle=False, **kwargs):
    """sklearn train_test_split wrapper."""
    return train_test_split(X, y, test_size=test_size, shuffle=shuffle, **kwargs)

def split_ts_data(X_ts_train, y, val_sdate, ndx_lvl=0):
    """Split time series training data at validation start date."""
    if ndx_lvl == 0:
        X_train = X_ts_train.loc[:val_sdate]
        X_val = X_ts_train.loc[val_sdate:]
        y_train = y.loc[:val_sdate]
        y_val = y.loc[val_sdate:]
        return X_train, X_val, y_train, y_val
    elif ndx_lvl == 1:
        X_train = X_ts_train.loc[:,:val_sdate,:]
        X_val = X_ts_train.loc[:,val_sdate:,:]
        y_train = y.loc[:val_sdate]
        y_val = y.loc[val_sdate:]
        return X_train, X_val, y_train, y_val


################### --- PREPROCESSING --- ###################

### FEATURE ENGINEERING ###
def make_lags(ts, lags, lead_time=1, col_idx_name=None):
    """Generate lag features to a time series Series."""
    return pd.concat(
        {
            f'y_lag_{i}': ts.shift(i)
            for i in range(lead_time, lags + lead_time)
        },
        axis='columns', names=col_idx_name)

def make_leads(ts, leads, name='lead', col_idx_name=None):
    """Generate lead features to a time series Series."""
    return pd.concat(
        {
            f'y_{name}_{i}': ts.shift(-i)
            for i in range(leads, 0, -1)
        },
        axis='columns', names=col_idx_name)


### FORMATING ###
def si_prefix(number, unit):
    """Convert a value into specified (SI system) prefix-unit representation.
    
    Parameters:
        number (int/float): number to be converted 
        unit (str): physical unit to be used
        
    Returns: (str)
    """
    
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


################### --- MODELLING --- ###################
    """Split time series training data at validation start date."""
    if ndx_lvl == 0:
        X_train = X_ts_train.loc[:val_sdate]
        X_val = X_ts_train.loc[val_sdate:]
        y_train = y.loc[:val_sdate]
        y_val = y.loc[val_sdate:]
        return X_train, X_val, y_train, y_val
    elif ndx_lvl == 1:
        X_train = X_ts_train.loc[:,:val_sdate,:]
        X_val = X_ts_train.loc[:,val_sdate:,:]
        y_train = y.loc[:val_sdate]
        y_val = y.loc[val_sdate:]
        return X_train, X_val, y_train, y_val        
    
####### PREDICTING #######

def return_fit_test(X_train, X_test, y, model, clip=False):
    """Return model y_fit and y_forecast as DF."""
    
    y_fit = pd.DataFrame(model.predict(X_train), 
                         index=X_train.index,
                         columns=y.columns)
    y_fore = pd.DataFrame(model.predict(X_test),
                          index=X_test.index,
                          columns=y.columns)
    if clip:
        return y_fit.clip(.0), y_fore.clip(.0)
    else:
        return y_fit, y_fore


####### METRICS #######
    
def rmsle(y_true, y_pred, **kwargs):
    return mean_squared_log_error(y_true, y_pred, **kwargs) ** 0.5
    
def rmsle_scores(y_train, y_val, y_fit, y_pred) -> None:
    rmsle_train = mean_squared_log_error(y_train, y_fit) ** 0.5
    rmsle_valid = mean_squared_log_error(y_val, y_pred) ** 0.5
    print(f'Training RMSLE: {rmsle_train:.5f}')
    print(f'Validation RMSLE: {rmsle_valid:.5f}')
    
    return None


def compare_scores(y_test, y_pred, worst=5, return_scores=False):
    """Compare predictions by family."""
    y_test = (y_test
              .stack(['store_nbr', 'family'])
              .reset_index()
              .copy()
              .rename({0:'sales'}, axis='columns')
    )
    y_pred = (y_pred
              .stack(['store_nbr', 'family'])
              .reset_index()
              .copy()
              .rename({0:'sales_pred'}, axis='columns')
    )
    
    y_test['sales_pred'] = y_pred.sales_pred.clip(0.)
    top_errors = (y_test
                  .groupby('family')
                  .apply(lambda x: rmsle(x['sales'], x['sales_pred']))
                  .sort_values(ascending=False)
    )
    if return_scores: 
        return top_errors
    else: 
        return top_errors.head(worst)
    

####### PARAMETER TUNING #######

def grid_search(X, y, estimator, param_grid, 
                scoring='rmsle', n_jobs=-1, cv=5, top=3):
    """GridSearchCV wrapper. Returns top 3 best scores/errors."""
    
    warnings.simplefilter("ignore", category=ConvergenceWarning)

    if scoring == 'rmsle':
        scoring=make_scorer(rmsle, greater_is_better=False)
    
    grid_search_cv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=scoring,
        n_jobs=n_jobs,
        cv=cv
    ).fit(X, y)
    
    # return the results
    result_df = (pd.DataFrame(grid_search_cv.cv_results_)
        .sort_values('rank_test_score')
        .set_index('rank_test_score')
        .loc[:, ['params', 'mean_test_score', 'std_test_score']]
    )
    # rename index
    result_df.index.name = 'rank'
    
    # show 2 floating points
    result_df['params'] = result_df.params.map(
        lambda x: {k:round(v, 5) for k,v in x.items()})
    
    # show score as positive
    result_df['mean_test_score'] = \
        result_df['mean_test_score'].map(lambda x: abs(x))
    
    return result_df.head(top)

