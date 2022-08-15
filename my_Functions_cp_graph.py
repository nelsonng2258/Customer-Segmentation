#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:18:45 2022

@author: nelsonng
"""

# Import libraries. 
import matplotlib.pyplot as plt 
import numpy as np 
import seaborn as sns 

# ---------------------------------------------------------------------------------------------- 

# Standard sub functions for plotting graphs. 
def set_rcparams(): 
    
    '''Set rcParams setting for graph layout.'''
    
    # Reset layout to default. 
    plt.rcdefaults()  
    
    # Set rcParams settings.
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Times New Roman' 
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600 


def set_sns_font_dpi(): 
    
    '''Set sns.setting for graph font and dpi.'''
    
    # Reset layout to default. 
    plt.rcdefaults() 
     
    # Set sns.set settings. 
    sns.set_style({'font.serif':'Times New Roman'}) 
    sns.set(rc = {"figure.dpi":600, 'savefig.dpi':600}) # Improve dpi. 


def set_sns_large_white(): 
    
    '''Set sns.setting to create a graph with large and whitegrid.'''
    
    # Reset layout to default. 
    plt.rcdefaults()  

    # Set style to times new roman.
    sns.set(rc = {'figure.figsize':(15, 6)}) # width x height. # 15, 10
    sns.set_style('whitegrid') # Set background.  

# ---------------------------------------------------------------------------------------------- 

def null_barchart(df, df_txt):

    '''Display bar chart to identify missing null values.'''
    
    # Reset layout to default. 
    set_rcparams() 
    
    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 
    
    # Plot bar chart. 
    df.isna().sum().plot(kind="bar", color='red')
    
    # Title of the bar chart. 
    plt.title('Bar Chart of Null values\n(' + str(df_txt) + ')', size=20)  
    
    # Display the bar chart. 
    plt.show() 


def null_heatmap(df, df_txt, fig_wid, fig_ht, ttl_siz):
    
    '''Display heatmap to identify missing null values.'''
    
    # Set graph layout. 
    set_rcparams() 
    
    # Set figure layout.  
    plt.subplots(figsize=(fig_wid, fig_ht)) # width x height
    
    # Colors for heatmap. 
    colours = ['springgreen', 'red'] # springgrean is non-null, red is null. 
    
    # Plot the heatmap. 
    cols = df.columns  
    g = sns.heatmap(df[cols].isnull(), cmap=sns.color_palette(colours), cbar_kws={'label': 'Red=Null, Green=Non-Null', 'orientation': 'vertical'})
    
    # Set title for the heatmap. 
    g.set_title('Heatmap of Non-null and Null Values\n(' + str(df_txt) + ')', size=ttl_siz)
    
    # Display the heatmap. 
    plt.show() 


def h_boxplot(df, col):
    
    '''Create a horizontal boxplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi()
    set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height

    # Create boxplot. 
    sns.boxplot(x = df[col], orient = 'h', showmeans=True)
    
    # Set title. 
    plt.title('Box Plot\n' + col, size=20)
    
    # Display plot. 
    plt.show() 

# ---------------------------------------------------------------------------------------------- 

# Subfunction of print_mult_countplot func. 
def hue_countplot(df, x, hue):
    
    '''Plot countplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()   

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 

    if hue:
        
        # Plot countplot.
        g = sns.countplot(data=df, x=x, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot scatterplot. 
        g = sns.countplot(data=df, x=x) 

    # Set title. 
    ttl_txt = 'Countplot\n' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_countplot func.  
def countplot_x_hue_lt(x_lt, hue_lt): 
    
    '''Create a list of tuples of x, y and hue values for print_mult_countplot func.'''

    x_hue_lt = [] 

    # Iterating over nested list. 
    for i in x_lt:
        for j in hue_lt:
            
            # Append x, hue values into x_hue_lt. 
            x_hue_lt.append((i, j))
                    
    return x_hue_lt

def print_mult_countplot(df, x_lt, hue_lt): 
    
    '''Print multiple countplot.'''
    
    # Iterate over countplot_x_hue_lt.    
    for i in countplot_x_hue_lt(x_lt, hue_lt):
    
        # Create variables x, hue from i.
        x = i[0] 
        hue = i[1] 
        
        # Print title. 
        ttl_txt = '\033[1m' + x + ' (' + hue + ')'+ '\033[0m'
        print(ttl_txt)
        
        # Plot countplot. 
        hue_countplot(df, x, hue)

        # Display countplot.
        plt.show()    

# ---------------------------------------------------------------------------------------------- 

# Subfunction of print_mult_catplots func.  
def hue_stripplot(df, x, y, hue):
    
    '''Plot stripplot.''' 
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()   

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 
    
    # Segregating based on presence of hue.
    if hue:
        
        # Plot stripplot.
        g = sns.stripplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot stripplot.
        g = sns.stripplot(data=df, x=x, y=y, size=2) 

    # Set title. 
    ttl_txt = 'Stripplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()
    
# Subfunction of print_mult_catplots func.  
def hue_swarmplot(df, x, y, hue):
    
    '''Plot swarmplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()  

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height  
    
    # Segregating based on presence of hue.
    if hue:
        
        # Plot swarmplot.
        g = sns.swarmplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot swarmplot.
        g = sns.swarmplot(data=df, x=x, y=y, size=2) 

    # Set title. 
    ttl_txt = 'Swarmplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_catplots func.  
def hue_boxplot(df, x, y, hue):
    
    '''Plot boxplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()  

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 
    
    # Segregating based on presence of hue.
    if hue:
        
        # Plot boxplot.
        g = sns.boxplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot boxplot.
        g = sns.boxplot(data=df, x=x, y=y) 

    # Set title. 
    ttl_txt = 'Boxplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_catplots func.  
def hue_violinplot(df, x, y, hue):
    
    '''Plot violinplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()   

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 
    
    # Segregating based on presence of hue.
    if hue:
        
        # Plot violinplot.
        g = sns.violinplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot violinplot.
        g = sns.violinplot(data=df, x=x, y=y) 

    # Set title. 
    ttl_txt = 'Violinplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_catplots func.  
def hue_boxenplot(df, x, y, hue):
    
    '''Plot boxenplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi() 
    set_sns_large_white()  

    # Set figure layout.  
    plt.subplots(figsize=(15, 6)) # width x height 
    
    # Segregating based on presence of hue.
    if hue:
        
        # Plot boxenplot.
        g = sns.boxenplot(data=df, x=x, y=y, hue=hue)

        # Location for legend to be placed. 
        g.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), ncol=2, title=hue)

    else:
        
        # Plot boxenplot.
        g = sns.boxenplot(data=df, x=x, y=y) 

    # Set title. 
    ttl_txt = 'Boxenplot\n' + y + ' VS ' + x
    plt.title(str(ttl_txt), size=20)  

    # Rotate the xticks to 90 degree.
    plt.xticks(rotation = 90)

    # Display plot.
    plt.show()

# Subfunction of print_mult_catplots func.  
def catplots_x_y_hue_lt(x_lt, y_lt, hue_lt): 
    
    '''Create a list of tuples of x, y and hue values for print_mult_catplots func.'''

    x_y_hue_lt = [] 

    # Iterating over nested list. 
    for i in x_lt:
        for j in y_lt:
            for k in hue_lt:
                    
                # x and y cannot be equal.  
                if i != j: 
                    
                    # Append x, y, hue values into x_y_hue_lt. 
                    x_y_hue_lt.append((i, j, k))
                    
    return x_y_hue_lt

def print_mult_catplots(df, x_lt, y_lt, hue_lt): 
    
    '''Print multiple stripplot, swarmplot, boxplot, violinplot, boxenplot.'''
    
    # Iterate over catplots_x_y_hue_lt.   
    for i in catplots_x_y_hue_lt(x_lt, y_lt, hue_lt):
    
        # Create variables x, y, hue from i.
        x = i[0] 
        y = i[1]
        hue = i[2] 
        
        # Print title. 
        ttl_txt = '\033[1m' + y + ' VS ' + x + ' (' + hue + ')' + '\033[0m'
        print(ttl_txt)
        
        # Plot stripplot, swarmplot, boxplot, violinplot, boxenplot. 
        hue_stripplot(df, x, y, hue)
        hue_swarmplot(df, x, y, hue)
        hue_boxplot(df, x, y, hue)
        hue_violinplot(df, x, y, hue)
        hue_boxenplot(df, x, y, hue)

        # Display stripplot, swarmplot, boxplot, violinplot, boxenplot.
        plt.show()         

# ---------------------------------------------------------------------------------------------- 

def corr_matrix(df):
    
    '''Plot correlation matrix.'''
    
    #  Set figure layout.  
    set_sns_font_dpi()
    set_sns_large_white()

    # Calculate correlation matrix.
    corr = df.corr()

    # Create mask to remove excess blocks.
    mask = np.triu(np.ones_like(corr, dtype=bool))
    
    # Customized cmap. 
    cmap = sns.diverging_palette(h_neg=10, h_pos=240, as_cmap=True)
    
    # Create heatmap.
    g = sns.heatmap(df.corr(), mask=mask, center=0, cmap=cmap, linewidths=1, annot=True, fmt='.2f')

    # Set title.
    g.set_title('Correlation Matrix', size=20)

    # Display plot.
    plt.show() 


def jointplot(df, X, y):
    
    '''Plot joinplot.'''
    
    # Set graph layout. 
    set_sns_font_dpi()
    set_sns_large_white() 

    # Modify figure size.
    plt.figure(figsize=(15,3)) # width x height

    # kde plot of RecodedCluster and Income
    g = sns.jointplot(x=X, y=y, data=df, kind='kde')
    
    # Set title.
    ttl_txt = 'Jointplot\n(' + str(y) + ' VS ' + str(X) + ')' 
    g.fig.suptitle(ttl_txt, y=1.07, size=20)
    
    # Display plot. 
    plt.show() 