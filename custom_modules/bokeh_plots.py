# -*- coding: utf-8 -*-
"""
Python file for accessing custom made bokeh plots.
"""
import pandas as pd

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, HoverTool, BoxSelectTool, \
    NumeralTickFormatter
from bokeh.palettes import Category20

### SCATTER PLOT ###
def scatter(df, x, y, label=''):
    '''Plots bokeh scatter plot.
    Parameters
    ----------
    df : dict / DataFrame
         Dictionary or pandas DataFrame where plotting data resides.
    x : str
         Key or column name for data in x-axis.
    y : str
         Key or column name for data in y-axis.      
    '''
    
    #init the bokeh nativbe source object
    source = ColumnDataSource(df)
    
    #init the figure/canvas for the plot
    p = figure(plot_height=350)
    
    #create the glyphs on canvas
    p.circle(x, y, size=10, color="navy", alpha=0.5, source=source,
             hover_fill_color='red', selection_fill_color='red', legend_label=label)
    
    #labels
    p.title.text = x + ' vs '+ y
    p.xaxis.axis_label = x
    p.yaxis.axis_label = y

    #axis properties
    p.yaxis[0].formatter = NumeralTickFormatter(format='0,0')
    p.xaxis[0].formatter = NumeralTickFormatter(format='0,0')
    
    #legend properties
    if label=='':
        p.legend.visible = False
    else:
        pass
    
    #adding tools
    hover = HoverTool(tooltips=[(y, "@"+y), (x, "@"+x)])
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
    #feature class values sorted alphabetically
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
    
    out = groups.apply(outliers).dropna().reset_index() #multindex (class, range index) series
    
    #sort outlier index values by mean class mean value
    out = pd.merge(left=q_50, right=out, how='outer', on=x, 
                   suffixes=('_mean', '')).sort_values(
                   by=y+'_mean').set_index(x)
    
    #construct outlier coordinates (if outliers excist)
    if not out.empty:
        outx = out.index.values #class names for x coordinates
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
    df = pd.DataFrame(data={'classes':classes, 'q_25':q_25[y], 'q_50':q_50[y],'q_75':q_75[y],
                            'upper':upper[y], 'lower':lower[y], 'color':colors}).sort_values(
                            by='q_50', ascending=False)
    
    #creating the bokeh source obj
    source = ColumnDataSource(df)
    
    #creating the canvas
    p = figure(plot_height=450, x_range=df.classes, 
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

