# -*- coding: utf-8 -*-
"""
Python file for accessing custom made bokeh plots.
"""
import pandas as pd
import numpy as np
from scipy.stats import norm, linregress
 

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, \
    NumeralTickFormatter, Range1d, LinearAxis, Legend
from bokeh.palettes import Category20

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

### BOX PLOT ###
def box(df, x, y):
    """Plots bokeh box plot.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame with all necessary data.
    x : str
        DataFrame column name for the x axis.
    y : str
        DataFrame column name for the y axis.
        
    Returns
    -------
    None
    """

    #string class values sorted alphabetically
    #and numeric class values ascendingly
    classes = sorted(df[x].unique())
    
    #group data by feature class values, returns DataFrameGroupBy object.
    groups = df[[x,y]].groupby(x)
    
    #find the quartiles and IQR for each class in the categorical data.
    #returned values are df-s
    q_25 = groups.quantile(q=0.25)
    q_50 = groups.quantile(q=0.5)
    q_75 = groups.quantile(q=0.75)
    iqr = q_75 - q_25
    upper = q_75 + 1.5*iqr
    lower = q_25 - 1.5*iqr

    #find the outliers for each class
    def outliers(group):
        class_name = group.name
        return group[(group[y] > upper.loc[class_name][y]) |
                     (group[y] < lower.loc[class_name][y])][y]
    
    #multindex (class, range index) series
    out = groups.apply(outliers).dropna().reset_index() 

    #sort outlier index values by mean class mean value
    out = pd.merge(left=q_50, right=out, how='outer', on=x, 
                   suffixes=('_mean', '')).sort_values(
                   by=y+'_mean').set_index(x)
    
    #construct outlier coordinates (if outliers excist)
    if not out.empty:
        outx = out.index.values.astype('str') #class names for x coordinates
        outy = out[y] #y values
    
    #if no outliers, shrink lengths of stems to be 
    #no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    lower[y] = [max([x2,y2]) for (x2,y2) in zip(list(qmin.loc[:,y]), lower[y])]
    upper[y] = [min([x2,y2]) for (x2,y2) in zip(list(qmax.loc[:,y]), upper[y])]
    
    #defining colors for x number of classes
    if len(classes) == 0 or len(classes) == 1:
        colors = ['blue']
    if len(classes) == 2:
        colors = ['blue','green']
    else:
        colors = Category20[len(classes)]
    
    #create df of the data for the boxes
    box_df = pd.DataFrame(data={'classes':classes, 'q_25':q_25[y], 'q_50':q_50[y],'q_75':q_75[y],
                            'upper':upper[y], 'lower':lower[y], 'color':colors}).sort_values(
                            by='q_50', ascending=False)
    
    #force 'class' dtype to be str
    box_df['classes'] = box_df['classes'].astype(str)
    
    #creating the bokeh source obj
    source = ColumnDataSource(box_df)
    
    #creating the canvas
    p = figure(plot_height=400, x_range=box_df.classes.unique(), 
               title=x+" vs "+y, x_axis_label=x,
               y_axis_label=y)
    
    #stems
    p.segment(x0='classes', y0='lower', x1='classes', y1='q_25', 
              line_color='black', source=source, name='seg')
    p.segment(x0='classes', y0='upper', x1='classes', y1='q_75', 
              line_color='black', source=source, name='seg')
    
    #boxes
    boxes = p.vbar(x='classes', bottom='q_25', top='q_75', width=0.7, 
                   line_color='black',color='color', source=source, name='box',
                   hover_fill_color='maroon')
    
    #add hover to show the mean value
    p.add_tools(HoverTool(renderers=[boxes], tooltips=[('Mean', '@q_50{0,0}')], 
                          names=['box'], point_policy='snap_to_data'))
    #means
    p.rect(x='classes', y='q_50', width=0.7, height=0.02, 
           source=source, color='black', name='mean')
    
    #whiskers
    p.rect(x='classes', y='lower', width=0.2, height=0.01, 
           line_color='black', source=source, name='rect')
    p.rect(x='classes', y='upper', width=0.2, height=0.01, 
           line_color='black', source=source, name='rect')
    
    source2 = ColumnDataSource(data={'outx':outx, 'outy':outy})
    
    #outliers
    if not out.empty:
        plot_out = p.circle(x='outx', y='outy', size=9, source=source2,
                            color='grey', fill_alpha=0.6, name='outl',
                            hover_fill_color='maroon', hover_line_color='maroon')
        p.add_tools(HoverTool(renderers=[plot_out], tooltips=[(y,'@outy')],
                              names=['outl'], point_policy='snap_to_data'))
    
    #figure props
    p.xgrid.visible = False
    p.ygrid.visible = False
    p.yaxis[0].formatter = NumeralTickFormatter(format='0,0')

    show(p)

### HISTOGRAM ###
def hist(df, feature, bins=50):
    '''Plots bokeh histogram, PDF & CDF of a DF feature.
    
    Parameters
    ----------
    df : DataFrame
        DF of the data.
    feature :  str
        Column name of the df.
    bins : int
        Number of bins to plot.
        
    Returns
    -------
    None
    '''
    
    #not nan feature values
    x = df[feature][df[feature].notna()].values 
    
    #Get the values for the histogram and bin edges (length(hist)+1)/
    #Use density to plot pdf and cdf on the same plot.
    hist, edges = np.histogram(x, bins=bins, density=True)
    
    ### PDF & CDF ##
    
    #find normal distribution parameters
    mu, sigma = norm.fit(x)
    xs = np.linspace(min(x), max(x)+1, len(x)) #x values to plot the line(s)
    
    pdf = norm.pdf(xs, loc=mu, scale=sigma) #probability distribution function
    cdf = norm.cdf(xs, loc=mu, scale=sigma) #cumulative distribution function
    
    #data sources for cdf
    source_cdf = ColumnDataSource({'cdf':cdf, 'xs':xs})
    
    #create the canvas
    p1 = figure(title='Histogram, PDF & CDF', plot_height=400,
                x_axis_label=feature, y_axis_label='Density')
    
    #add histogram
    p1.quad(bottom=0, top=hist, left=edges[:-1], right=edges[1:],
          fill_color='royalblue', line_color='black', alpha=0.7)
    
    #add pdf
    p1.line(xs, pdf, line_color='red', line_width=5, 
            alpha=0.5, legend_label='PDF')
    
    #set left-hand y-axis range
    p1.y_range = Range1d(0, max(hist) + 0.05*max(hist))
    
    #setting the second y axis range name and range
    p1.extra_y_ranges = {"cdf": Range1d(start=0, end=1.05)}
    
    #adding the second y axis to the plot and to the right.  
    p1.add_layout(LinearAxis(y_range_name="cdf", axis_label='CDF'), 'right')

    #add cdf with y range on the right
    cdf_plot = p1.line('xs', 'cdf', source=source_cdf, alpha=0.8, 
                       line_color='darkgoldenrod', line_width=5, 
                       legend_label='CDF', y_range_name='cdf', name='cdf',
                       hover_line_color='green')
    
    #hover tool
    p1.add_tools(HoverTool(renderers=[cdf_plot], tooltips=[('Prob', '@cdf{0.00}')],
                           mode='hline'))
    
    #figure properties
    p1.xgrid.visible = False
    
    #hide entries when clocking on a legend
    p1.legend.click_policy="hide"
    
    show(p1)

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

