# -*- coding: utf-8 -*-
"""
Python file for accessing custom made bokeh plots.
"""
import pandas as pd
import numpy as np
import scipy as sp

from scipy import stats
from scipy.stats import norm, linregress, skew, skewtest, pearsonr

import pandas_bokeh

from statsmodels.tsa.stattools import pacf

from bokeh.models import ColumnDataSource, CDSView, GroupFilter, \
    HoverTool, RangeTool, BoxSelectTool, \
    Range1d, LinearAxis, Legend, LegendItem, Label, \
    NumeralTickFormatter, DatetimeTickFormatter

from bokeh.io import curdoc
from bokeh.plotting import figure, show
from bokeh.layouts import column, row, gridplot
from bokeh.palettes import Category10, Category20, Plasma, Viridis256

### LINE PLOT ###
def line(df, x, y, height=350, width=700, x_axis_type='auto'):
    """Plots bokeh line plot.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with all necessary data.
    x : str
        DataFrame column name for the x axis.
    y : str
        DataFrame column name for the y axis.
    """
    
    #create an informative df
    new_df = df.groupby(x)[y].mean().to_frame() #mean
    new_df = new_df.join(df.groupby(x)[y].median(), rsuffix='med') #median
    new_df = new_df.join(df.groupby(x)[x].count()) #count
    new_df.columns = ['mean', 'median', 'count']
    new_df = new_df.reset_index()

    source = ColumnDataSource(new_df)
    
    p = figure(height=height, width=width, title=x + ' vs ' + y,
               x_axis_label=x, y_axis_label=y, x_axis_type=x_axis_type)
    
    #initialize colors
    colors = ['#008000', '#0000FF']
    
    #capture legend items
    legend_items = []
    
    for col,color in zip(new_df.columns.values[1:-1], colors):
        
        #create circles
        circles = p.circle(x=x, y=col, source=source, size=6, 
                           hover_fill_color='maroon', color=color, alpha=0.1)
        #create line
        lines = p.line(x=x, y=col, source=source, line_width=2, 
                       color=color)

        #add hover tool
        p.add_tools(HoverTool(renderers=[circles], 
                              tooltips=[('x', '@'+x),('y', '@'+col+'{0,0}'),('count', '@count')]))

        #add legend items to a list
        legend_items.append((col, [circles,lines]))
        
    #y axis ticks
    p.yaxis[0].formatter = NumeralTickFormatter(format='0,0')
    
    ##LEGEND##
    legend = Legend(items=legend_items, location='center') #legend object

    p.add_layout(legend, place='right') #extra area to the figure

    #hide entries when clocking on a legend
    p.legend.click_policy="hide"

    show(p)


### SCATTER PLOT ###
def scatter(df, x, y, reg=False):
    '''Plots bokeh scatter plot.
    Parameters
    ----------
    df : dict / DataFrame
         Dictionary or pandas DataFrame where plotting data resides.
    x : str
         Key or column name for data in x-axis.
    y : str
         Key or column name for data in y-axis.
    reg : bool
        To plot regression line or not. Default False.
    
    Returns
    -------
    None
    '''
    
    #init the bokeh nativbe source object
    source = ColumnDataSource(df)
    
    #init the figure/canvas for the plot
    p = figure(plot_height=350)
    
    #create the glyphs on canvas
    circle = p.circle(x, y, size=10, color="navy", alpha=0.5, 
                      source=source,hover_fill_color='red', 
                      selection_fill_color='red')
    
    #add regression fit
    if reg == True:
        #find slope, intercept, rvalue, pvalue, stderr of the regression line
        slope, intercept, rvalue, pvalue, stderr = linregress(x=df[x], y=df[y])
        
        #plot regression fit
        p.line(x=df[x], y=intercept + slope * df[x], color='red', alpha=0.35,
               line_width=6, legend_label=r'r={}'.format(round(rvalue,2)))
        
    #labels
    p.title.text = x + ' vs '+ y
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y

    #axis properties
    p.yaxis[0].formatter = NumeralTickFormatter(format='0,0')
    p.xaxis[0].formatter = NumeralTickFormatter(format='0,0')
        
    #adding tools
    hover = HoverTool(renderers=[circle], tooltips=[(y, "@"+y), (x, "@"+x)])
    p.add_tools(hover, BoxSelectTool())
    
    # show the results
    show(p)


### SCATTER + REGRESSION PLOT ###
def regplot(df: pd.DataFrame, 
            x: str, 
            y: str,
            reg: bool=True,
            alpha: float=0.95,
            extra_hover_tooltips: [tuple]=None,
            hover_formatters: dict={},
            show_figure=True,
            **fig_kwargs):
    '''Plots bokeh scatter plot.
    Parameters
    ----------
    df : dict / DataFrame
         Dictionary or pandas DataFrame where plotting data resides.
    x : str
         Key or column name for data in x-axis.
    y : str
         Key or column name for data in y-axis.
    reg : bool, default True
        To plot regression line or not. Default False.
    alpha : float, default 0.95
        Probability that random variable will be drawn from the returned 
        range. Confidence interval.
    extra_hover_tooltips : list of tuples, default None
        List of hover tool tooltips of the form [('name', '@name{}'), ... ,]
    hover_formatters : dict, default None
        Dictionary of hover tool tooltip formatters of the form 
        {'name': 'datetime'}.
        
    Returns
    -------
    Figure : bokeh.figure, default None
        IIf show_figure is True returns None (shwos the figure), otherwise
        returns figure to which axis is connected.'''
    
    df = df.reset_index()
    source = ColumnDataSource(df) # bokeh source object
    
    # init the figure/canvas for the plot
    fig = figure(**fig_kwargs, 
                 title=f"{x} vs {y}",
                 x_axis_label=x, 
                 y_axis_label=y)
    
    # add data to figure
    circle = fig.circle(x, y, size=10, alpha=0.75, source=source, 
                        hover_fill_color='goldenrod',
                        hover_line_color='goldenrod',
                        selection_fill_color='goldenrod')
    
    # synthetic x values for regression lines
    xs = np.linspace(df[x].min(), df[x].max() + 1, num=len(df[x]))
    colors = ['red', 'green']
    orders = ['1st Order', '2nd Order']
    legend_items = []
    
    # add regression lines and confidence intervals
    if reg: 
        for order, name, color in zip([1,2], orders, colors):
            # calculate Confidence Interval
            ci_95 = norm.interval(
                alpha=alpha,            # confidence interval
                loc=0,                  # mean of the distribution
                scale=stats.sem(df[y])) # standard error of the mean

            # find polynomial coefficients, highest degree to lowest
            coefs = np.polyfit(x=df[x], y=df[y], deg=order)
            ys = coefs[0] * xs + coefs[1] if order == 1 else \
                 (coefs[0] * xs ** 2) + (coefs[1] * xs) + coefs[2]

            # plot regression line and confidence intervals
            reg_line = fig.line(x=xs, y=ys, 
                                color=color, alpha=0.75, line_width=5)

            # plot regression confidnece intervals
            ci = fig.varea(x=xs, y1=ys + ci_95[0], y2=ys + ci_95[1],
                           alpha=0.25, color=color)
            
            # calculate Pearson correlation coefficient
            r = pearsonr(df[x] ** order, df[y])[0]
            
            # add legend items
            legend_items.append(LegendItem(label=f"{name} r = {r:.2f}", 
                                           renderers=[reg_line, ci]))
    # add hover tool
    hover = HoverTool(
        renderers=[circle], 
        tooltips=[(y, f"@{y}"), (x, f"@{x}")],
        formatters=hover_formatters)
    
    if extra_hover_tooltips is not None:
        hover.tooltips.append(*extra_hover_tooltips)
    
    # add tools and legend to figure
    fig.add_tools(hover, BoxSelectTool())
    fig.add_layout(Legend(click_policy='hide', items=legend_items))
    
    if show_figure:
        return show(fig)
    else:
        return fig


### BOX PLOT ###
def box(df: pd.DataFrame, 
        x: str, 
        y: str, 
        sort_by: str=None,
        ascending: bool=True,
        title: str=None,
        xlabel: str=None,
        ylabel: str=None,
        y_axis_format: str='0,0',
        hover_mean_format: str= '0.0',
        height: int=400,
        show_figure: bool=True,
        **fig_kwargs):
    """Plots bokeh box plot to show distributions with respect to categories.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all necessary data.
    x : str
        DataFrame column name for the x axis.
    y : str
        DataFrame column name for the y axis.
    sort_by : str, default None
        If None the resulting boxes are sorted by mean value. Boxes are sorted
        by provided DataFrame column name. 
    ascending : bool, default True
        Wether to sort categories ascendingly or descendigly when sort == True.
    title : str, default 'Distribution of y by x'
        Title of the figure.
    x_label : str, default x
        X axis label.
    y_label : str, default y
        Y axis label.
    y_axis_format : str, default '0,0'
        Bokeh NumeralTickFormatter number format.
    hover_mean_format : str, default '0.0'
        Bokeh HoverTool tooltip format.
    height : int, default 400
        Figure height in pixels.
    show_figure : bool, default True
        Weather to show figure or return the bokeh.figure axis.   
    fig_kwargs : key-word arguments
        Key-word arguments for main figure. E.g. width, height, title.
    Returns
    -------
    fig : bokeh.figure
    """
    
    # string class values sorted alphabetically numeric ascendingly
    classes = sorted(df[x].unique())
    
    # group data by feature class values, returns DataFrameGroupBy object.
    groups = df[[x,y]].groupby(x)
    
    # find the quartiles and IQR for each class in the categorical data.
    # returned values are df-s
    q_25 = groups.quantile(q=0.25)
    q_50 = groups.quantile(q=0.5)
    q_75 = groups.quantile(q=0.75)
    iqr = q_75 - q_25
    upper = q_75 + 1.5*iqr
    lower = q_25 - 1.5*iqr

    # find the outliers for each class
    def outliers(group):
        class_name = group.name
        return group[(group[y] > upper.loc[class_name][y]) |
                     (group[y] < lower.loc[class_name][y])][y]
    
    # multindex (class, range index) series
    out = groups.apply(outliers).dropna().reset_index() 

    # sort outlier index values by mean class mean value
    out = pd.merge(left=q_50, right=out, how='outer', on=x, 
                   suffixes=('_mean', '')).sort_values(
                   by=y+'_mean').set_index(x)
    
    # construct outlier coordinates (if outliers excist)
    if not out.empty:
        outx = out.index.values.astype('str') # class names for x coordinates
        outy = out[y] # y values
    
    # if no outliers, shrink lengths of stems to be 
    # no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    lower[y] = [max([x2,y2]) for (x2,y2) in zip(list(qmin.loc[:,y]), lower[y])]
    upper[y] = [min([x2,y2]) for (x2,y2) in zip(list(qmax.loc[:,y]), upper[y])]

    # defining colors for n number of classes
    n = len(classes) 
    if n <= 1:
        colors = ['blue']
    if n == 2:
        colors = ['blue','green']
    if n <= 20:
        colors = Category20[n]
    else:
        # select colors uniformly along the spectrum to have variance
        every_nth = 256 // n
        colors = [Viridis256[i] for i in np.arange(0, every_nth * n, 10)]
    
    # create df of the data for the boxes
    box_df = pd.DataFrame(data={
        'classes':classes, 'q_25':q_25[y], 'q_50':q_50[y],'q_75':q_75[y],
        'upper':upper[y], 'lower':lower[y], 'color':colors})
    
    # add sort_by column to data
    if sort_by is not None:
        box_df[sort_by] = box_df['classes'].apply(lambda i: df[df[x]==i][sort_by][0])

    # force 'class' dtype to be str
    box_df['classes'] = box_df['classes'].astype(str)
    
    # sort data by class values (alphabetically or ascendingly)
    box_df = box_df.sort_values(by=sort_by, ascending=ascending) \
             if sort_by is not None else box_df.sort_values(by='q_50', ascending=ascending)
    
    # creating the bokeh source obj
    source = ColumnDataSource(box_df)
    
    # creating the figure
    title = f"Distribution of {y} by {x}" if title is None else title
    xlabel = x if xlabel is None else xlabel
    ylabel = y if ylabel is None else ylabel
    
    fig = figure(x_range=box_df.classes.unique(), title=title, 
                 x_axis_label=xlabel, y_axis_label=ylabel, height=height,
                 **fig_kwargs)
    
    # box color specs based on dark or light background
    line_color = curdoc().theme._json['attrs']['Axis']['major_tick_line_color']
    
    # stems
    fig.segment(x0='classes', y0='lower', x1='classes', y1='q_25', 
              line_color=line_color, source=source, name='seg')
    fig.segment(x0='classes', y0='upper', x1='classes', y1='q_75', 
              line_color=line_color, source=source, name='seg')
    
    # boxes
    boxes = fig.vbar(x='classes', bottom='q_25', top='q_75', width=0.7, 
                     line_color=line_color, color='color', source=source, 
                     name='box',hover_fill_color='maroon')
    
    # means
    means = fig.rect(x='classes', y='q_50', width=0.7, height=0.02, 
                     source=source, color=line_color, name='mean',
                     hover_fill_color='blue')
    
    # add hover to show the mean value
    fig.add_tools(HoverTool(renderers=[boxes], 
                            tooltips=[('Mean', f'@q_50{{{hover_mean_format}}}')], 
                            names=['box'], point_policy='snap_to_data'))
    
    # whiskers
    fig.rect(x='classes', y='lower', width=0.2, height=0.01, 
           line_color=line_color, source=source, name='rect')
    fig.rect(x='classes', y='upper', width=0.2, height=0.01, 
           line_color=line_color, source=source, name='rect')
    
    source2 = ColumnDataSource(data={'outx':outx, 'outy':outy})
    
    # outliers
    if not out.empty:
        plot_out = fig.circle(x='outx', y='outy', size=6, source=source2,
            color='grey', fill_alpha=0.3, name='outl', legend_label='outliers',
            hover_fill_color='maroon', hover_line_color='maroon')
        
        plot_out.visible = False
        fig.add_tools(HoverTool(renderers=[plot_out], tooltips=[(y,'@outy')],
                                names=['outl'], point_policy='snap_to_data'))
    
    # clickable legend
    fig.legend.click_policy = 'hide'
    fig.legend.background_fill_alpha=0.75
    
    # figure props
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.yaxis[0].formatter = NumeralTickFormatter(format=y_axis_format)
    
    # show or return fig
    if show_figure: show(fig)
    else: return fig
    

### NORMALIZED HISTOGRAM ###
def density_hist(df : pd.DataFrame, 
                 feature : str, 
                 bins : int=50, 
                 plot_height : int=400, 
                 plot_width : int=700, 
                 **fig_kwargs):
    '''Plots numpy histogram with probability density function value at the
    bin, normalized such that the integral over the range is 1. Additionally
    plots PDF and Cumulative Density Function on the same figure.
    
    Parameters
    ----------
    df : pd.DataFrame
        DF of the data.
    feature :  str
        Column name of the df.
    bins : int, default 50
        Number of equally spaced bins to plot.
        
    Returns
    -------
    fig : bokeh.plotting.figure.Figure
    '''
    # not nan feature values
    x = df[feature][df[feature].notna()].values 
    
    # Get the values for the histogram and bin edges (length(hist)+1)/
    # Use density to plot pdf and cdf on the same plot.
    hist, edges = np.histogram(x, bins=bins, density=True)
    
    ### PDF & CDF ##
    
    # find normal distribution parameters
    mu, sigma = norm.fit(x)
    xs = np.linspace(min(x), max(x)+1, len(x)) #x values to plot the line(s)
    
    pdf = norm.pdf(xs, loc=mu, scale=sigma) 
    cdf = norm.cdf(xs, loc=mu, scale=sigma) 
    
    # data sources for cdf
    source_cdf = ColumnDataSource(
        {'pdf':pdf, 'cdf':cdf, 'xs':xs, 'cdf_pc':cdf * 100})
    
    # create the canvas
    fig = figure(title=f"{feature} distribution", plot_height=plot_height,
                 plot_width=plot_width,x_axis_label=feature, 
                 y_axis_label='Density', **fig_kwargs)
    
    # add histogram
    fig.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:],
             fill_color='royalblue', line_color='black', alpha=0.7,
             legend_label=feature)

    # add pdf plot and hovertool 
    pdf_plot = fig.line('xs', 'pdf', source=source_cdf, line_color='red', 
                       line_width=5, alpha=0.5, legend_label='PDF')
    
    fig.add_tools(HoverTool(renderers=[pdf_plot], 
                           tooltips=[('PDF', '@pdf')],
                           mode='vline'))
    
    # set left-hand y-axis range
    fig.y_range = Range1d(0, max(hist) * 1.05)
    
    # setting the second y axis range name and range
    fig.extra_y_ranges = {"cdf": Range1d(start=0, end=1.05)}
    
    # adding the second y axis to the plot and to the right.  
    fig.add_layout(
        LinearAxis(y_range_name="cdf", axis_label='CDF',
                   formatter=NumeralTickFormatter(format="0%")), 'right')
    
    # add cdf with y range on the right and hovertool
    cdf_plot = fig.line('xs', 'cdf', source=source_cdf, alpha=0.8, 
                       line_color='darkgoldenrod', line_width=5, 
                       legend_label='CDF', y_range_name='cdf', name='cdf')
    
    fig.add_tools(HoverTool(renderers=[cdf_plot], 
                           tooltips=[('CDF', '@cdf_pc{0.0}%')],
                           mode='vline'))

    # figure properties
    fig.xgrid.visible = False
    
    # hide entries when clicking on a legend
    fig.legend.click_policy="hide"
    
    # add figure below for statistics
    fig_stats = figure(width=fig.width, height=50, outline_line_alpha=1,
                       toolbar_location=None, x_range=[0,1], 
                       y_range=[0,1], tools='')
    
    # set the components of the stats figure invisible
    for fig_component in [fig_stats.grid[0], fig_stats.ygrid[0],
                          fig_stats.xaxis[0], fig_stats.yaxis[0]]:
        fig_component.visible = False
    
    # statistics to test normality
    skew_test = skewtest(df[feature])
    skew_ = skew(df[feature])
    
    # annotation texts
    skewtest_label = Label(x=0.5, y=0.5, text_color='#FFFFFF', 
        render_mode='canvas', text_font_size='14px',
        text="Skewtest:", text_align='center')
    
    skew_zscore = Label(x=0.1, y=0.05, text_color='#FFFFFF', 
        render_mode='canvas', text_font_size='13px',
        text=f"z-score: {skew_test[0]:.2f}", text_align='left')
    
    skew_pvalue = Label(x=0.5, y=0.05, text_color='#FFFFFF', 
        render_mode='canvas', text_font_size='13px',
        text=f"p-value: {skew_test[1]:.2f}", text_align='center')
    
    skew_value = Label(x=0.9, y=0.05, text_color='#FFFFFF', 
        render_mode='canvas', text_font_size='13px',
        text=f"skew: {skew_:.2f}", text_align='right')
    
    # add annotations to the plot
    fig_stats.add_layout(skewtest_label)
    fig_stats.add_layout(skew_zscore)
    fig_stats.add_layout(skew_pvalue)
    fig_stats.add_layout(skew_value)
    
    return show(column([fig, fig_stats]))


### DOUBLE HISTOGRAM ###
def double_hist(dfs, feature, names=['train', 'test'], bins=50):
    """Plotting 2 histograms side by side.
    
    Parameters
    ----------
    dfs : list of DataFrames
        List of DataFrames.
    feature : str
        Feature name in df.
    names : list of strings
        Names for the dataframes.
    bins : int
        Number of bins per histogram.
        
    Returns
    -------
    None
    """
    
    import numpy as np
    
    #find joint max range of two dataframes
    max_value = max([dfs[0][feature].max(), dfs[1][feature].max()]) 
    min_value = min([dfs[0][feature].min(), dfs[1][feature].min()])
    
    #creating numpys histograms in the background so it
    #divides the data into bins and returns bin edges
    hist1, edges1 = np.histogram(dfs[0][feature], bins=bins, 
                                 range=[min_value,max_value], density=True)
    hist2, edges2 = np.histogram(dfs[1][feature], bins=bins, 
                                 range=[min_value,max_value], density=True)
    
    #calculating step value for separating 
    #two histogram bins side by side
    step = (edges1[1]-edges1[0]) * 0.5
    
    # create the empty canvas
    p = figure(plot_height = 400, x_axis_label=feature, y_axis_label='Density',
               title = 'Histogram of ' + names[0] + ' & ' + names[1] + 'sets')
    
    # Add a quad glyph
    p.quad(bottom=0, top=hist1, 
           left=edges1[:-1], right=edges1[1:]-step, 
           fill_color='blue', line_color='black', legend_label=names[0])
    p.quad(bottom=0, top=hist2, 
           left=edges2[:-1]+step, right=edges2[1:], 
           fill_color='green', line_color='black', legend_label=names[1])
    
    #legend position
    p.legend.location = "top_right"
    
    #figure props
    p.xgrid.visible = False
    
    show(p)


### TIME-SERIES ###
def calendar(
    df : pd.DataFrame,
    ys : [str],
    exclude_values_dict : dict={}, 
    include_values_dict : dict={},
    ylabel : str=None,
    hover_format : str=None,
    show_figure : bool=True,
    **fig_kwargs):
    """Plot (multi)line time series with other informative features. Bool dtypes
    are included by default.
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with all data to be plotted with datetime index dtype.
    ys : list, [str]
        List of column names to be plotted as time series y values. First 
        column name in the list is used for y_values for other columns.
    exclude_values_dict : dict, default {}
        Dictionary of the form {column:[value_1, ... ,value_n] to be excluded
        from plotting.
    include_values_dict : dict, default {}
        Dictionary of the form {column:[value_1, ... ,value_n] to be included
        to the resulting plot.
    ylabel : str, default None
        Y-label for time series y axis.
    hover_format : str, default None
        Format of the hover tool datetime values. 
    show_figure : bool, default True
        Weather to show figure or return the bokeh.figure axis.
    fig_kwargs : key-word arguments
        Key-word arguments for main figure. E.g. width, height, title.
        
    Returns
    -------
    fig : bokeh.figure"""
    
    # dictionaries to contain different column names
    len_exclude_dict = len(exclude_values_dict)
    len_include_dict = len(include_values_dict)
    if len_exclude_dict > 0 and len_include_dict > 0:
        if len(set(exclude_values_dict.keys()) & 
               set(include_values_dict.keys())) != 0:
            raise ValueError(f"Dictionaries can't contain same columns!")
    
    x = df.index.name                           # index name
    source = ColumnDataSource(df.reset_index()) # create bokeh source
    
    if ylabel == None: ylabel=ys[0]             # y-axis label
        
    # MAIN FIGURE #
    fig = figure(
        y_axis_label=ylabel,
        x_axis_type='datetime',
        **fig_kwargs)
    
    start_index = int(0.75 * len(source.data[x])) # explicitly set initial
    start = source.data[x][start_index]           # range for the figure
    end = source.data[x][-1]
    fig.x_range = Range1d(start, end)
    
    # RANGETOOL FIGURE #
    fig_rangetool = figure(
        title='Range Tool',
        height=130, 
        width=fig.width, 
        y_range=fig.y_range,
        x_axis_type='datetime',
        y_axis_type=None,
        tools="",
        toolbar_location=None,
    )
    
    range_tool = RangeTool(x_range=fig.x_range)
    range_tool.overlay.fill_color = "navy"
    range_tool.overlay.fill_alpha = 0.4

    fig_rangetool.ygrid.grid_line_color = None
    fig_rangetool.add_tools(range_tool)
    fig_rangetool.toolbar.active_multi = range_tool
    
    # x-axis tick format
    x_range_delta = df.index[-1] - df.index[0]  # data range in days
    dt_formatter = DatetimeTickFormatter(
        milliseconds=["%H:%M:%S.%f"],
        seconds=["%H:%M:%S"],
        minutes=["%H:%M:%S"],
        hours=["%H:%M:%S"],
        days=["%d %B %Y"],
        months=["%d %B %Y"],
        years=["%d %B %Y"],
    )
    if x_range_delta <= pd.Timedelta(366, unit='D'):
        dt_formatter.days = ["%b"]
        dt_formatter.months = ["%b"]
        dt_formatter.years = ["%Y"]
        dt_hover_format = "%d %b"
        
    dt_hover_format = dt_hover_format if hover_format is None else hover_format 
    fig.xaxis.formatter = dt_formatter
    fig_rangetool.xaxis.formatter = dt_formatter
    
    # DATA #
    bools = df.select_dtypes(bool).columns
    features = [*ys, *bools, *exclude_values_dict.keys(), 
                *include_values_dict.keys()]
    n_features = len(features) # number of features
    # determine glyph colors
    palette = Category10 if n_features < 11 else Category20
    colors = ['blue', 'orange'] if n_features < 3 else palette[n_features]
    
    legend_items = []
    all_renderers = []
    for name, color in zip(features, colors):
        # init hovertool for each glyph on main
        hover = HoverTool( 
            tooltips=[(x, f"@{x}{{{dt_hover_format}}}")],
            formatters={f"@{x}":'datetime'},
            mode='vline')
        
        glyph_main = None
        glyph_range = None
        if name in ys:
            # draw glyphs on mian and on range tool
            glyph_main = fig.line(x=x, y=name, source=source, 
                                  color=color, line_width=2)
            glyph_range = fig_rangetool.line(x=x, y=name, source=source, 
                                             color=color)
            # add glyphs to renderers
            renderers = [glyph_main, glyph_range]
            all_renderers += renderers
            
            hover.tooltips.append((name, f"@{name}"))
            legend_items.append(LegendItem(
                label=f" {name}", renderers=renderers))
            
        else: 
            cds = None
            if name in bools: 
                # query True values only 
                true_idx = df[name][df[name] == True].index
                temp_df = df[ys[0]].loc[true_idx].reset_index()
                cds = ColumnDataSource(temp_df)
            
            elif name in exclude_values_dict.keys(): # values to exclude
                idx = df[name][~df[name].isin(exclude_values_dict[name])] \
                    .index
                cds = ColumnDataSource(df.loc[idx,[ys[0],name]].reset_index())
            
            else: # values to include
                idx = df[name][df[name].isin(include_values_dict[name])] \
                    .index
                cds = ColumnDataSource(df.loc[idx,[ys[0],name]].reset_index())
            
            # create glyphs
            glyph_main = fig.circle(x=x, y=ys[0], source=cds, 
                color=color, size=8, alpha=0.5)
            glyph_range = fig_rangetool.circle(x=x, y=ys[0], 
                source=cds, color=color, size=4, alpha=0.5)
            
            # add glyps to rednerers list
            renderers = [glyph_main, glyph_range]
            all_renderers += renderers
            
            # hover and legend
            hover_string = "True" if name in bools else f"@{name}"
            hover.tooltips.append((name, hover_string))
            legend_items.append(LegendItem(
                label=f" {name}", 
                renderers=renderers))
        
        hover.renderers = [glyph_main] # for HoverTool
        all_renderers += renderers # for legend
        
        fig.add_tools(hover)

    # Dummy fig for legend
    fig_legend = figure(width=130, height=fig.height + 130, 
                        outline_line_alpha=0,toolbar_location=None,
                        border_fill_color='#ffffff')
    
    # set the components of the figure invisible
    for fig_component in [fig_legend.grid[0], fig_legend.ygrid[0],
                          fig_legend.xaxis[0], fig_legend.yaxis[0]]:
        fig_component.visible = False
    
    # set the figure range outside of the range of all glyphs
    fig_legend.renderers += all_renderers
    fig_legend.x_range.end = fig.x_range.end + pd.Timedelta(365, unit='D')
    fig_legend.x_range.start = fig.x_range.start + pd.Timedelta(360, unit='D')
    fig_legend.add_layout(Legend(click_policy = "hide", location='center', 
                                 items=legend_items, border_line_width=2))
    
    if show_figure:
        return show(row(column(fig,fig_rangetool), fig_legend))
    else:
        return row(column(fig,fig_rangetool), fig_legend)

def periodogram(ts: [pd.Series, pd.DataFrame],
                 show_figure: bool=True,
                 **fig_kwargs):
    """Plot bokeh periodogram wrapped around scipy.signal.periodogogram.
    
    Parameters
    ----------
    ts : pd.Series or pd.Dataframe
        Pandas series or DataFrame with all columns to be plotted.
    show_figure : bool, default True
        Weather to show figure or return the bokeh.figure axis.
    fig_kwargs : key-word args
        Key-word arguments for plot_bokeh figure.
    Returns
    -------
    None
    """
    # periodogram nice xticks
    period_coefs = [1, 2, 4, 6, 12, 26, 52, 104] 
    periods = ["Annual (1)",
               "Semiannual (2)",
               "Quarterly (4)",
               "Bimonthly (6)",
               "Monthly (12)",
               "Biweekly (26)",
               "Weekly (52)",
               "Semiweekly (104)"]
    periods_dct = {coef:period for coef,period in zip(period_coefs, periods)}
    
    # convert ts to dataframe if series
    df = ts.to_frame() if isinstance(ts, pd.Series) else ts
    
    # determine sampling frequency
    sampling_f = pd.Timedelta('365D') / pd.Timedelta('1D')
    
    temp_df = pd.DataFrame() # init empty df
    for col in df.columns:
        # create periodogram profile(s)
        frequencies, spectrum = sp.signal.periodogram(
            x=df[col],
            fs=sampling_f,
            window='boxcar',
            detrend='linear',
            scaling='spectrum')
    
        if temp_df.empty: 
            temp_df.index = frequencies # add x if not already
        temp_df[col] = spectrum # add spectrum per column
    
    temp_df.index.name = 'period' # name the index
    
    # create period categorical names
    temp_df['period_cat'] = "Semiweekly"
    for i, period in enumerate(period_coefs):
        if i == len(period_coefs) - 1: 
            break
        start = 0 if i == 0 else (period_coefs[i-1] + period) / 2
        next_period = period_coefs[i+1]
        end = (next_period + period_coefs[i+2]) / 2 if i != 6 else 78
        temp_df.loc[start:end, 'period_cat'] = periods_dct[period][:-4]
    
    source = ColumnDataSource(temp_df.reset_index()) # init source
    
    # define pallette
    n_cols = len(temp_df.columns)
    colors = ['purple', 'goldenrod']
    pallette = Category20[n_cols] if n_cols > 2 else \
               (colors[0] if n_cols == 1 else colors)
    
    # create the figure
    fig = figure(title="Periodogram",
                 x_axis_label="Period",
                 y_axis_label='Variance',
                 x_axis_type="log",
                 **fig_kwargs)
    
    # remap to nice xticks and rotate labels to fit
    fig.xaxis.ticker = period_coefs
    fig.xaxis.major_label_overrides = periods_dct
    fig.xaxis.major_label_orientation = np.pi / 6
    
    # create graphs and store renderers and legend items
    all_renderers = []
    legend_items = []
    for name, color in zip(temp_df.columns, pallette):
        # add step glyphs
        step = fig.step(x='period', y=name, source=source, color=color, 
                        line_width=4, mode="before")
        
        circle = fig.circle(x='period', y=name, source=source, size=10,
                            alpha=0)
        renderers = [step, circle]
        hover = HoverTool(tooltips=[(name ,f"@{name}"), ("period", "@period_cat")], 
                          renderers=[circle], point_policy='snap_to_data', 
                          mode='vline')
        fig.add_tools(hover)
        
        legend_items.append(LegendItem(label=f"{name}", renderers=renderers))
        all_renderers += renderers
    
    # add legend
    fig.add_layout(Legend(click_policy='hide', items=legend_items))
    
    if show_figure:
        show(fig)
    else:
        return fig


def correlogram(x: [[], np.array, pd.Series, pd.DataFrame], 
                lags: int, 
                zero: bool=False, 
                show_figure: bool=True, 
                **fig_kwargs):
    
    """Plot n specified autocorrelation lags. Based on 
    statsmodels.tsa.stattools.pacf estimate.
    
    Parameters
    ----------
    x : array-like
        Observations of time series for which pacf is calculated.
    lags : int
        Number of lags to return autocorrelation for. 
    zero : bool, default False
        Flag indicating whether to include the 0-lag autocorrelation.
    show_figure : bool, default True
        Weather to show figure or return the bokeh.figure axis.
    kwargs** : keyword arguments, optional
        Optional keyword arguments of bokeh.plotting.figure.Figure. E.g. 
        'plot_width', 'plot_height'. 
        
    Returns
    -------
    Figure : figure, None, default None
        If show_figure is True returns None (shwos the figure), otherwise
        returns figure to which axis is connected."""
    
    # lags as x-values
    xs = [*np.arange(0, lags+1)] if zero else [*np.arange(1, lags+1)]
    
    # partial autocorrelation values as ys and confidence intervals
    auto_corrs, ci = pacf(x=x, nlags=lags, alpha=.05) 
    
    # remove first 0th lag
    if not zero:
        auto_corrs = auto_corrs[1:]
        ci = ci[1:]
    
    ci = [arr - pa for arr, pa in zip(ci, auto_corrs)] # confidence intervals
    ci_cutoff = abs(ci[0][0])
    
    # prepare the data into df
    df = pd.DataFrame({'x': xs, 'y': auto_corrs})
    df['high_low'] = df['y'].apply(lambda x: 'high' if abs(x) >= ci_cutoff else 'low')
    df['xs_multi'] = df['x'].apply(lambda x: [x, x])
    df['ys_multi'] = df['y'].apply(lambda x: [0, x])
    
    # create CDS and Views
    source = ColumnDataSource(df)
    view_low = CDSView(source=source, 
        filters=[GroupFilter(column_name='high_low', group='low')])
    view_high = CDSView(source=source, 
        filters=[GroupFilter(column_name='high_low', group='high')])
    
    # init main figure
    fig = figure(**fig_kwargs, title='Partial Autocorrelation', 
                 x_axis_label=f'{x.name} Lags', y_axis_label='Partial Correlation')
    
    fig.x_range = Range1d(0, lags + 1) # explicitly set initial range
    baseline = fig.line(x=[-10, lags + 10], y=[0, 0], line_width=3)
    
    legend_items = []
    hover_renderers = []
    all_renderers = []
    for view_, type_, name_, size_ in zip(
        [view_low, view_high], ['circle', 'star'], 
        ['Insignificant','Significant'], [5, 9]):
        
        # draw vertical lines and circles
        multi_line = fig.multi_line(
            xs='xs_multi',
            ys='ys_multi',
            source=source,
            view=view_,
            line_color='#1f77b4',
            line_width=3,
            hover_line_color='goldenrod')
    
        # draw circles or stars
        glyphs = fig.scatter(x='x', y='y', source=source, view=view_, 
                             size=size_, color='#1f77b4', marker=type_,
                             hover_fill_color='goldenrod', 
                             hover_line_color='goldenrod')
        
        # store renderers
        renderers = [multi_line, glyphs]
        legend_items.append(LegendItem(label=f"{name_}", renderers=renderers))
        hover_renderers += [multi_line]
        all_renderers += renderers
        
    # add HoverTool    
    hover = HoverTool(renderers=hover_renderers,
                          tooltips=[('PACF', "@y{0.00}"),('Lag', "@x")],
                          line_policy='interp')
    fig.add_tools(hover)
        
    # draw 95% confidence intervals
    ci = fig.varea(x=xs, 
                   y1=[i[0] for i in ci],
                   y2=[i[1] for i in ci],
                   alpha=0.10)
    
    legend_items.append(LegendItem(label="95% CI", renderers=[ci]))
    fig.add_layout(Legend(click_policy='hide', items=legend_items))
    
    if show_figure:
        show(fig)
    else:
        return fig

def lag_plot(x: pd.Series, 
             y=None,
             lag: int=1, 
             lead: int=None, 
             standardize: bool=False,
             show_figure: bool=False,
             **fig_kwargs):
    """Plot n-th lag or lead scatter plot with regression fit.
    
    Parameters
    ----------
    x : pd.Series
        Pandas series of a feature to be plotted.
    y : array-like, default None"
        Pre-computed lag or lead array of values.
    lag: int, default 1
        Shift x values n-units forward.
    lead : int, default None
        Shift x values n-units backward.
    standardize : bool, default False
        Wether to normalize the data or not.
    show_figure: bool, default False
        Return figure axis or show the figure.
    fig_kwargs : key-word args
        Figure key-word arguments like: plot_height, plot_width,
    
    Returns
    -------
    Figure : figure, None, default None
        If show_figure is True returns None (shwos the figure), otherwise
        returns figure to which axis is connected."""
    
    # check both lag and lead are not None
    if lag == lead == None:
        raise ValueError("Both lag and lead can't be None!")
    elif type(lag) == type(lead) == int:
        raise ValueError("Both lag and lead can't be specified!")
    
    # create lag or lead
    type_ = 'lag'
    number_ = lag
    if lead is not None:
        x_ = x.shift(-lead)
        type_ = 'lead'
        number_ = lead
    else:
        x_ = x.shift(lag)
    
    # standardize if specified
    if standardize:
        x_ = (x_ - x_.mean()) / x_.std()
    if y is not None:
        y_ = (y - y.mean()) / y.std() if standardize else y
    else:
        y_ = x
    
    x_ = x_.dropna() # drop NaN-s
    x_, y_ = x_.align(y_, join='left')
    
    # init main figure
    fig = figure(
        title=f"{type_.capitalize()} {number_}", 
        y_axis_label=f"{x.name}", 
        x_axis_label=f"{x.name}_{type_}_{number_}",
        **fig_kwargs)
    
    source = ColumnDataSource({f"{x.name}{type_}":x_, x.name:y_})
    
    # plot scatter
    scatter = fig.circle(x=f"{x.name}{type_}", y=x.name, source=source, 
                         size=7, hover_fill_color='goldenrod',
                         hover_line_color='goldenrod')
    
    # plot regression fit
    slope, intercept, rvalue, pvalue, stderr = linregress(x=x_, y=y_)
    
    fig.line(x=x_, y=slope * x_ + intercept, color='red', alpha=0.75,
             line_width=5, legend_label=f"r = {rvalue:.2f}")
    
    # add hovertool
    hover = HoverTool(
        renderers=[scatter],
        tooltips=[(f"{x.name}", f"@{x.name}"),
                  (f"{type_}", f"@{x.name}{type_}")])
    
    fig.add_tools(hover)
    
    # hide entries when clicking on a legend
    fig.legend.click_policy="hide"
    
    if show_figure:
        return show(fig)
    else:
        return fig    

def plot_lags(x: pd.Series, 
              y=None,
              lags: [int, [int]]=None,
              leads: [int, [int]]=None,
              ncols: int=4,
              **grid_kwargs):

    
    # list of leads and lags actual values 
    leads_ = [*range(1,leads+1)] if type(leads) == int else leads
    lags_ = [*range(1,lags+1)] if type(lags) == int else lags
    
    all_plots = []
    if leads is not None:
        for lead in leads_:
            temp_fig = lag_plot(x=x, y=y, lag=None, lead=lead)
            all_plots.append(temp_fig)
    if lags is not None:
        for lag in lags_:
            temp_fig = lag_plot(x=x, y=y, lag=lag, lead=None)
            all_plots.append(temp_fig)
    
    grid = gridplot(all_plots, ncols=ncols, **grid_kwargs)
    show(grid)

    