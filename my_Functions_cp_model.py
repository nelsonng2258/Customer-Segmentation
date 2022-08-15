#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 12:35:05 2022

@author: nelsonng
"""

import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import seaborn as sns 

from sklearn.cluster import KMeans 
from sklearn.feature_selection import chi2, SelectKBest, VarianceThreshold  
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge, RidgeCV 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, mean_squared_error, precision_recall_curve, r2_score, roc_auc_score, roc_curve 
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler 

from statsmodels.formula.api import logit, probit 
from statsmodels.stats.outliers_influence import variance_inflation_factor 

# Import my packages. 
import my_Functions_cp_graph as myfcp_gra 

# ----------------------------------------------------------------------------------------------

def scaled_df(df):
    
    '''Create scaled dataframe.'''
    
    # Create scaler. 
    scaler = StandardScaler()
    
    # Fit scaler to dataframe. 
    scaler.fit(df)
    
    # Transform dataframe.
    scaled_df = scaler.transform(df)
    
    # Add columns to dataframe.
    scaled_df = pd.DataFrame(scaled_df, columns=df.columns)

    return scaled_df


def elbowplot(df, n_clus):
    
    '''Print elbowplot.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi() 
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height
    
    sse = []
    
    for i in range(1,n_clus+1):
        
        # Append sse for the respective number of clusters. 
        km = KMeans(n_clusters=i)
        km.fit(df)
        sse.append(km.inertia_)
    
    # Plot sse against number of clusters..
    plt.plot(list(range(1,n_clus+1)), sse, '-x', linewidth=2, markersize=10)
    
    # Set title.
    plt.title('Elbow Plot', size=20)
    
    # Set xlabel and ylabel. 
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Distance (SSE)')
    
    # Display plot. 
    plt.show()

# ----------------------------------------------------------------------------------------------

# Subfunction of clus3_df func. 
def swap_clus3_tup_lt(df):
    
    '''A list of tuples for swapping cluster groupings in descending order based on size.'''
    
    swap_clus3_tup_lt = []

    # Recode to cluster 3. 
    # Recode less than 380 data points to cluster 3 for current cluster 1. 
    if len(df[df['Cluster'] == 1]) < 380:

        # Append tuple (1,3).
        swap_clus3_tup_lt.append((1,3))

    # Recode less than 380 data points to cluster 3 for current cluster 2. 
    elif len(df[df['Cluster'] == 2]) < 380:

        # Append tuple (2,3).
        swap_clus3_tup_lt.append((2,3))

    # Recode less than 380 data points to cluster 3 for current cluster 3. 
    elif len(df[df['Cluster'] == 3]) < 380:

        # Append tuple (3,3).
        swap_clus3_tup_lt.append((3,3))

    # Recode to cluster 2. 
    # Recode more than 380 data points to less than 1000 data points to cluster 1 for current cluster 2. 
    if len(df[df['Cluster'] == 1]) > 380 and len(df[df['Cluster'] == 1]) < 1000:

        # Append tuple (1,2).
        swap_clus3_tup_lt.append((1,2))

    # Recode more than 380 data points to less than 1000 data points to cluster 2 for current cluster 2. 
    elif len(df[df['Cluster'] == 2]) > 380 and len(df[df['Cluster'] == 2]) < 1000:

        # Append tuple (2,2).
        swap_clus3_tup_lt.append((2,2))

    # Recode more than 380 data points to less than 1000 data points to cluster 3 for current cluster 2. 
    elif len(df[df['Cluster'] == 3]) > 380 and len(df[df['Cluster'] == 3]) < 1000:

        # Append tuple (3,2).
        swap_clus3_tup_lt.append((3,2))

    # Recode to cluster 1. 
    # Recode more than 1000 data points to cluster 1 for current cluster 1. 
    if len(df[df['Cluster'] == 1]) > 1000:

        # Append tuple (1,1).
        swap_clus3_tup_lt.append((1,1))

    # Recode more than 1000 data points to cluster 1 for current cluster 2. 
    elif len(df[df['Cluster'] == 2]) > 1000:

        # Append tuple (2,1).
        swap_clus3_tup_lt.append((2,1))

    # Recode more than 1000 data points to cluster 1 for current cluster 3. 
    elif len(df[df['Cluster'] == 3]) > 1000:

        # Append tuple (3,1).
        swap_clus3_tup_lt.append((3,1))
        
    return swap_clus3_tup_lt

def clus3_df(df_1, df_2):
    
    '''Create dataframe with 3 cluster groups.'''

    # Fit k-means. 
    X = df_1.values
    km = KMeans(n_clusters=3, init='k-means++', random_state=0) 
    y_clus = km.fit_predict(X)

    # Assign clusters to df.  
    df = df_2.copy() 
    df['Cluster'] = y_clus+1
    
    # Create a list of tuples to update current cluster group numbers. 
    f_swap_clus3_tup_lt = swap_clus3_tup_lt(df)

    rc_clus_lt = [] 
    
    # Iterate over column 'Cluster' of df dataframe. 
    for i in df['Cluster']:
        
        # Iterate over f_swap_clus3_lt. 
        for j in f_swap_clus3_tup_lt: 
            
            # Identify current cluster group number.
            if i == j[0]:
                
                # Append updated cluster group number.
                rc_clus_lt.append(j[1])
    
    # Update column 'Cluster' with update cluster group numbers.
    df['Cluster'] = rc_clus_lt 
    
    return df

# ---------------------------------------------------------------------------------------------- 

def print_clus_sz(df, clus_no):
    
    '''Print cluster size of the respective cluster.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height
    
    # Plot the cluster size.
    sns.countplot(x='Cluster', data=df)
    
    # Set title. 
    plt.title('Cluster Size of ' + str(clus_no) + ' Clusters\n(K-Means)', size=20) 
    
    # Display plot. 
    plt.show()
    

def clus2_df(df_1, df_2):
    
    '''Create dataframe with 2 cluster groups.'''

    # Fit k-means. 
    X = df_1.values
    km = KMeans(n_clusters=2, init='k-means++', random_state=0) 
    y_clus = km.fit_predict(X)

    # Assign clusters to df.  
    df = df_2.copy() 
    df['Cluster'] = y_clus+1
    
    clus_lt = []
    
    # Recode if cluster 1 is less than 1000 recode cluster numbers. 
    if len(df[df['Cluster'] == 1]) < 1000:
        
        # Iterate over 'Cluster' column of dataframe. 
        for i in df['Cluster']:
            
            # Recode to cluster 2. 
            if i == 1:
                clus_lt.append(2)
            
            # Recode to cluster 1. 
            else:
                clus_lt.append(1)
    
        # Update dataframe with new cluster numbers.
        df['Cluster'] = clus_lt 
    
    return df

# ----------------------------------------------------------------------------------------------

# Subfunction of print_clus_mult_catplot func. 
def clus_ttl_stripplot(df, cols, ttl): 
    
    '''Plot stripplot for cluster dataframe with title.''' 
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 

    # Modify figure size.
    fig = plt.figure(figsize=(15,3)) # width x height
    
    # Set title. 
    fig.suptitle(ttl, fontsize=20, y = 1.1)

    for i, j in enumerate(cols): 

        # Set subplots. 
        plt.subplot(1,len(cols),i+1)
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        # Plot stripplot.
        sns.stripplot(y='Cluster', x=j, data=df, orient='h')

        # Set ylabel.
        plt.ylabel('Cluster' if i==0 else '')
    
    # Display plot. 
    plt.show() 

# Subfunction of print_clus_mult_catplot func. 
def clus_swarmplot(df, cols):
    
    '''Plot swarmplot for cluster dataframe.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,3)) # width x height
    
    for i, j in enumerate(cols): 

        # Set subplots. 
        plt.subplot(1,len(cols),i+1)
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        # Plot swarmplot.
        sns.swarmplot(y='Cluster', x=j, data=df, orient='h', size=2)

        # Set ylabel.
        plt.ylabel('Cluster' if i==0 else '')

    # Display plot. 
    plt.show() 

# Subfunction of print_clus_mult_catplot func. 
def clus_boxplot(df, cols):
    
    '''Plot boxplot for cluster dataframe.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,3)) # width x height
    
    for i, j in enumerate(cols): 

        # Set subplots. 
        plt.subplot(1,len(cols),i+1)
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        # Plot boxplot.
        sns.boxplot(y='Cluster', x=j, data=df, orient='h')

        # Set ylabel.
        plt.ylabel('Cluster' if i==0 else '')
    
    # Display plot. 
    plt.show() 

# Subfunction of print_clus_mult_catplot func. 
def clus_violinplot(df, cols):
    
    '''Plot violinplot for cluster dataframe.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,3)) # width x height
    
    for i, j in enumerate(cols): 

        # Set subplots. 
        plt.subplot(1,len(cols),i+1)
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        # Plot violinplot.
        sns.violinplot(y='Cluster', x=j, data=df, orient='h')

        # Set ylabel.
        plt.ylabel('Cluster' if i==0 else '')
    
    # Display plot. 
    plt.show() 

# Subfunction of print_clus_mult_catplot func. 
def clus_boxenplot(df, cols):
    
    '''Plot boxenplot for cluster dataframe.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,3)) # width x height
    
    for i, j in enumerate(cols): 

        # Set subplots. 
        plt.subplot(1,len(cols),i+1)
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        # Plot boxenplot.
        sns.boxenplot(y='Cluster', x=j, data=df, orient='h')

        # Set ylabel.
        plt.ylabel('Cluster' if i==0 else '')
    
    # Display plot. 
    plt.show() 

def print_clus_mult_catplot(df, cols, ttl):
    
    '''Print multiple catplots from cluster dataframe of selected columns.'''
        
    # Plot cluster's stripplot.  
    clus_ttl_stripplot(df, cols, ttl)
    
    # Plot cluster's swarmplot.  
    clus_swarmplot(df, cols)
    
    # Plot cluster's boxplot. 
    clus_boxplot(df, cols)
    
    # Plot cluster's violinplot. 
    clus_violinplot(df, cols)
    
    # Plot cluster's boxenplot. 
    clus_boxenplot(df, cols) 

# ----------------------------------------------------------------------------------------------

# Subfunction of print_clus_mult_countplot func.  
def clus_ttl_countplot(df, cols, clus_no, ttl):
    
    '''Plot countplot with title for selected cluster in cluster dataframe.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    fig = plt.figure(figsize=(15,3)) # width x height
    
    # Set title. 
    fig.suptitle(ttl, fontsize=20, y = 1.1)
    
    for i, j in enumerate(cols): 

        # Set subplots. 
        plt.subplot(1,len(cols),i+1)
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        # Plot boxenplot.
        sns.countplot(x=j, data=df[df['Cluster'] == clus_no], orient='h')

        # Set ylabel.
        plt.ylabel('Cluster ' + str(clus_no) if i==0 else '')
    
    # Display plot. 
    plt.show() 

# Subfunction of print_clus_mult_countplot func. 
def clus_countplot(df, cols, clus_no):
    
    '''Plot countplot for selected cluster in cluster dataframe.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,3)) # width x height
    
    for i, j in enumerate(cols): 

        # Set subplots. 
        plt.subplot(1,len(cols),i+1)
        plt.subplots_adjust(hspace=0.3, wspace=0.2)
        
        # Plot boxenplot.
        sns.countplot(x=j, data=df[df['Cluster'] == clus_no], orient='h')

        # Set ylabel.
        plt.ylabel('Cluster ' + str(clus_no) if i==0 else '')
    
    # Display plot. 
    plt.show() 

def print_clus_mult_countplot(df, cols, ttl, clus_no):
    
    '''Print multiple countplots from cluster dataframe of selected columns.'''
    
    # Iterate over clus_no. 
    for i in range(clus_no):
        
        # Set cluster number. 
        clus_no = i + 1
        
        # Print title. 
        if i == 0:
            
            # Plot cluster's countplot with title. 
            clus_ttl_countplot(df, cols, clus_no, ttl)
        
        # Do not print title. 
        else:
            
            # Plot cluster's countplot without title. 
            clus_countplot(df, cols, clus_no)

# ----------------------------------------------------------------------------------------------

def print_high_corr(df, cols, corr_thold):
    
    '''Print features with high correlation.'''
    
    # Create positive correlation matrix. 
    corr_df = df[cols].corr().abs()

    # Create and apply mask. 
    mask = np.triu(np.ones_like(corr_df, dtype=bool)) 
    tri_df = corr_df.mask(mask)
    
    # Identity columns with high correlation.
    col_lt = [c for c in tri_df.columns if any(tri_df[c] > corr_thold)]
    
    # Print title.
    print('\033[1m' + 'High Correlation Features (Correlation = ' + str(corr_thold) + ')' + '\033[0m')
    
    # Print high correlation list. 
    for i in col_lt:
        print(i)
        

def vif_df(df, cols):
    
    '''Create dataframe for vif.''' 
    
    # Create indpendent variables set. 
    X = df[cols]

    # Create VIF dataframe. 
    vif_df = pd.DataFrame()
    vif_df['Feature'] = X.columns 

    # Calculate VIF for each feature. 
    vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    # Sort vif_data. 
    vif_df = vif_df.sort_values(by='VIF', ascending=False)  
    
    return vif_df        


def print_low_var(df, cols, var_thold): 
    
    '''Print features with low variance.'''
    
    # Select variance threshold.
    sel = VarianceThreshold(threshold=var_thold)
    
    # Fit and normalize. 
    sel.fit(df[cols]/df[cols].mean())
    
    # Create mask to create reduced_df. 
    mask = sel.get_support()
    reduced_df = df[cols].loc[:, mask] 

    # Identity columns with low variance.
    col_lt = [i for i in df[cols] if i not in reduced_df.columns]  
    
    # Print title.
    print('\033[1m' + 'Low Variance Features (Variance = ' + str(var_thold) + ')' + '\033[0m')
    
    # Print low variance list. 
    for i in col_lt:
        print(i)


def rc_clus2_lt(df, col): 
    
    '''Create a recoded cluster list for 2 clusters.'''
    
    rc_clus2_lt = []
    
    # Iterate over the selected column of dataframe.
    for i in df[col]:
        
        # Recode cluster 1 to 1. 
        if i == 1:
            rc_clus2_lt.append(1) 
        
        # Recode cluster 2 to 0. 
        else:
            rc_clus2_lt.append(0) 
        
    return rc_clus2_lt  


def print_sel_k_best(X, y, feat_no): 
    
    '''Print top features based on selectkbest.'''
    
    # Apply SelectKBest class to extract top features.
    bestfeatures = SelectKBest(score_func=chi2, k=2)
    fit = bestfeatures.fit(X,y)

    # Create dataframes.
    col_df = pd.DataFrame(X.columns)
    sc_df = pd.DataFrame(fit.scores_)

    # Concat columns. 
    sc_df = pd.concat([col_df, sc_df],axis=1)

    # Add column names to the columns. 
    sc_df.columns = ['Specs','Score'] 

    # Print top features. 
    print('\033[1m' + 'Top ' + str(feat_no) + ' Features for ' + str(y.columns[0]) + ' (SelectKBest)' + '\033[0m')
    print(sc_df.nlargest(feat_no,'Score')) 


def sel_k_best_df(X, y, feat_no): 
    
    '''Dataframe of top features based on selectkbest.'''
    
    # Apply SelectKBest class to extract top features.
    bestfeatures = SelectKBest(score_func=chi2, k=2)
    fit = bestfeatures.fit(X,y)

    # Create dataframes.
    col_df = pd.DataFrame(X.columns)
    sc_df = pd.DataFrame(fit.scores_)

    # Concat columns. 
    sc_df = pd.concat([col_df, sc_df],axis=1)

    # Add column names to the columns. 
    sc_df.columns = ['Specs','Score'] 
    
    # Sort sc_df based on 'Score' column. 
    sc_df.sort_values(by='Score', ascending=False, inplace=True)
    
    return sc_df


def print_vif_var(df_1, df_2, vif_val):
    
    '''Print a list of variables with VIF < vif_val.'''
    
    # Print title. 
    print('\033[1m' + 'List of variables with VIF < ' + str(vif_val) +':' + '\033[0m')
    
    # Iterate over df_1 column 'Specs'.
    for i in df_1['Specs'].values:
        
        # Iterate over df_2 column 'VIF' for VIF < 5.
        if i in df_2[df_2['VIF'] < vif_val]['Feature'].values: 
            
            # Print common variables of df_1 column 'Specs' and df_2 column 'VIF' for VIF < 5.
            print(i)


def ts_siz_1_2_alloc(prop_1, prop_2):
    
    '''Allocate test size 1 and 2 based on proportions.'''
    
    # Calculate test size 1 and 2. 
    ts_siz_1 = prop_1
    ts_siz_2 = prop_2/(1-prop_1)
    
    return ts_siz_1, ts_siz_2 

# ----------------------------------------------------------------------------------------------

# Subfunction of print_data_amt, linreg_reg_r2mean_r2std_msemean_msestd_cv_tup func, and linreg_reg_coef_intcp_rtrain_rval_rtest_msetrain_mseval_msetest_tup func. 
def linreg_train_valid_test_split(X, y, ts_siz_1, ts_siz_2):
    
    '''Create train, validation and test data for linear regression data.''' 
    
    # Perform temp-hold split.  
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, 
                                                      test_size=ts_siz_1, 
                                                      random_state=0) 
    
    # Perform train-valid split. 
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp,
                                                      test_size=ts_siz_2, 
                                                      random_state = 0)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def print_data_amt(X, y, ts_siz_1, ts_siz_2): 
    
    '''Print data amount for train, validate and test sets.'''
    
    # Perform train-valid-test split. 
    X_train, X_val, X_test, y_train, y_val, y_test = linreg_train_valid_test_split(X, y, ts_siz_1, ts_siz_2)
    
    # Calculate amount of data. 
    amt_train = len(X_train)
    amt_val = len(X_val)
    amt_test = len(X_test) 
    total_amt = amt_train + amt_val + amt_test
    
    # Calculate the percentage of type of data to total available data. 
    pct_amt_train = amt_train/total_amt
    pct_amt_val = amt_val/total_amt 
    pct_amt_test = amt_test/total_amt
    
    # Print amount of data and their percentages with respect to total data.
    print('\033[1m' + "Model Data Amount" + '\033[0m')
    
    amt_train_txt = 'Train Set: ' + str(amt_train) + ' ' + str(round(pct_amt_train,2)) + '%'
    amt_val_txt = 'Validate Set: ' + str(amt_val) + ' ' + str(round(pct_amt_val,2)) + '%'
    amt_test_txt = 'Test Set: ' + str(amt_test) + ' ' + str(round(pct_amt_test,2)) + '%'
    
    print(amt_train_txt)
    print(amt_val_txt)
    print(amt_test_txt) 

# ----------------------------------------------------------------------------------------------

def train_df(X, y, ts_siz_1, ts_siz_2):
    
    '''Create dataframe for train set.'''
    
    # Perform train test split. 
    X_train, X_val, X_test, y_train, y_val, y_test = linreg_train_valid_test_split(X, y, ts_siz_1, ts_siz_2)

    # Create dataframes for train data set.
    train_df = X_train.join(y_train)

    return train_df


def val_df(X, y, ts_siz_1, ts_siz_2):
    
    '''Create dataframe for validate set.'''
    
    # Perform train test split. 
    X_train, X_val, X_test, y_train, y_val, y_test = linreg_train_valid_test_split(X, y, ts_siz_1, ts_siz_2)

    # Create dataframes for validate data set.
    val_df = X_val.join(y_val)

    return val_df


def test_df(X, y, ts_siz_1, ts_siz_2):
    
    '''Create dataframe for test set.'''
    
    # Perform train test split. 
    X_train, X_val, X_test, y_train, y_val, y_test = linreg_train_valid_test_split(X, y, ts_siz_1, ts_siz_2)

    # Create dataframes for test data set.
    test_df = X_test.join(y_test)

    return test_df


def log_auc_plot(df, thold_val):
    
    '''Plot auc score of logit model.'''

    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 

    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height
    
    # Create dependent variable.
    y_data = df[df.columns[-1]]
    
    thold_lt = []
    auc_lt = [] 
    
    # Iterate over range 100.
    for i in [*range(100)]:
        
        # Set threshold value.
        thold = (i+1)*0.01
        
        # Find y predict value based on threshold value.
        y_pred = log_y_pred_arr(df, thold) 
        
        # Append threshold value into thold_lt. 
        thold_lt.append(thold) 
        
        # Append auc value into auc_lt. 
        auc_lt.append(roc_auc_score(y_data, y_pred))
    
    # Create dataframe for threshold and auc. 
    df = pd.DataFrame(auc_lt, thold_lt)
    
    # Reset index. 
    df.reset_index(inplace=True)
    
    # Rename columns. 
    columns = {df.columns[0]: 'Threshold', 
               df.columns[1]: 'AUC'} 
    df.rename(columns=columns, inplace=True)
    
    # Plot line plot. 
    sns.lineplot(x='Threshold', y='AUC', data=df)
    
    # Vertical line for threshold set. 
    plt.axvline(x=thold_val, color='black') 
    
    # Set title. 
    plt.title('Logit Model', size=20)
    
    # Display plot.
    plt.show() 
    
    
def pro_auc_plot(df, thold_val):
    
    '''Plot auc score of probit model.'''

    thold_lt = []
    auc_lt = []
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 

    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height
    
    # Create dependent variable.
    y_data = df[df.columns[-1]]
    
    # Iterate over range 100.
    for i in [*range(100)]:
        
        # Set threshold value.
        thold = (i+1)*0.01
        
        # Find y predict value based on threshold value.
        y_pred = pro_y_pred_arr(df, thold)
        
        # Append threshold value into thold_lt. 
        thold_lt.append(thold) 
        
        # Append auc value into auc_lt. 
        auc_lt.append(roc_auc_score(y_data, y_pred))
    
    # Create dataframe for threshold and auc. 
    df = pd.DataFrame(auc_lt, thold_lt)
    
    # Reset index. 
    df.reset_index(inplace=True)
    
    # Rename columns. 
    columns = {df.columns[0]: 'Threshold', 
               df.columns[1]: 'AUC'} 
    df.rename(columns=columns, inplace=True)
    
    # Plot line plot. 
    sns.lineplot(x='Threshold', y='AUC', data=df)
    
    # Vertical line for threshold set. 
    plt.axvline(x=thold_val, color='black') 
    
    # Set title. 
    plt.title('Probit Model', size=20)
    
    # Display plot.
    plt.show() 

# ----------------------------------------------------------------------------------------------

# Subfunction of log_precision_recall_curve_plot func and pro_precision_recall_curve_plot func. 
def inp_txt(df): 
    
    '''Create input text for logit and probit models.'''
    
    # Create str format for dependent variable and independent variables. 
    dv = df.columns[-1]
    iv = df.columns[:-1]

    # Add ' + ' between every independent variables.
    iv = ' + '.join(iv)

    # Add ' ~ ' between dependent variable and independent variables. 
    inp_txt = dv + str(' ~ ') + iv

    return inp_txt

# Subfunction of mult_precision_recall_curve_plot func. 
def precision_recall_vs_threshold_plot(precisions, recalls, thresholds, thold_val):
    
    '''Plot precision against thresholds and recall against thresholds.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Plot precision against thresholds. 
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    
    # Plot recall against thresholds. 
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    
    # Vertical line for threshold set. 
    plt.axvline(x=thold_val, color='black')  
    
    # Set legend location.
    plt.legend(loc="upper left")
    
    # Set xlabel.
    plt.xlabel("Threshold")
    
    # Set title. 
    plt.title("Precisions/ Recalls Trade off") 

# Subfunction of log_precision_recall_curve_plot func and pro_precision_recall_curve_plot func.
def mult_precision_recall_curve_plot(y_data, y_pred, thold_val): 
    
    '''Plot precision and recall curves.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Set precisions, recalls and thresholds. 
    precisions, recalls, thresholds = precision_recall_curve(y_data, y_pred)
    
    # Plot precision/ recall trade off. 
    plt.figure(figsize=(15, 12)) 
    plt.subplot(2, 2, 1)
    precision_recall_vs_threshold_plot(precisions, recalls, thresholds, thold_val)
    
    # Plot precision and recall curve. 
    plt.subplot(2, 2, 2)
    plt.plot(precisions, recalls)
    plt.xlabel("Precision")
    plt.ylabel("Recall") 
    plt.title("PR Curve: Precisions Recall Trade off")
    
    # Display plot. 
    plt.show() 

def log_precision_recall_curve_plot(df, thold_val):
    
    '''Plot precision and recall curves for logit model.'''
    
    # Create input text for dependent variable and independent variables. 
    f_inp_txt = inp_txt(df)
    
    # Fit logistic regression model. 
    mdl = logit(f_inp_txt, data=df).fit()
    
    # Create independent variables. 
    X_data = df[df.columns[:-1]] 
    
    # Create dependent variable.
    y_data = df[df.columns[-1]]
    
    # Predict y values based on X. 
    y_pred = mdl.predict(X_data)
    
    # Plot precision recall curves. 
    mult_precision_recall_curve_plot(y_data, y_pred, thold_val) 


def pro_precision_recall_curve_plot(df, thold_val):
    
    '''Plot precision and recall curves for probit model.'''
    
    # Create input text for dependent variable and independent variables. 
    f_inp_txt = inp_txt(df)
    
    # Fit logistic regression model. 
    mdl = probit(f_inp_txt, data=df).fit()
    
    # Create independent variables. 
    X_data = df[df.columns[:-1]] 
    
    # Create dependent variable.
    y_data = df[df.columns[-1]]
    
    # Predict y values based on X. 
    y_pred = mdl.predict(X_data)
    
    # Plot precision recall curves. 
    mult_precision_recall_curve_plot(y_data, y_pred, thold_val)

# ----------------------------------------------------------------------------------------------

# Subfunction of print_log_clf_report func and print_pro_clf_report func. 
def print_clf_report(y_data, y_pred, ttl):
    
    '''Print classification report.'''
    
    # Create clf_report in dataframe format. 
    clf_report = pd.DataFrame(classification_report(y_data, y_pred, output_dict=True))
    
    # Print classification report output. 
    print('\033[1m' + str(ttl)+ ' Result' + '\033[0m') 
    print('================================================')
    print(f'Accuracy Score: {accuracy_score(y_data, y_pred) * 100:.2f}%') 
    print(f'AUC Score: {roc_auc_score(y_data, y_pred):.2f}')
    print('_______________________________________________')
    print(f'CLASSIFICATION REPORT:\n {clf_report}')
    print('_______________________________________________') 
    print(f'Confusion Matrix:\n {confusion_matrix(y_data, y_pred)}\n')

def print_log_clf_report(df, thold_val, ttl):
    
    '''Print logit model classification report.''' 
    
    # Create dependent variable.
    y_data = df[df.columns[-1]]
    
    # Predict y values based on X. 
    y_pred = log_y_pred_arr(df, thold_val) 
    
    # Print classification report. 
    print_clf_report(y_data, y_pred, ttl)


def print_pro_clf_report(df, thold_val, ttl):
    
    '''Print probit model classification report.''' 
    
    # Create dependent variable.
    y_data = df[df.columns[-1]]
    
    # Predict y values based on X. 
    y_pred = pro_y_pred_arr(df, thold_val) 
    
    # Print classification report. 
    print_clf_report(y_data, y_pred, ttl) 

# ----------------------------------------------------------------------------------------------

def log_y_pred_arr(df, thold_val):
    
    '''Create an array of y prediction for each data point for logit model based on threshold value.'''
    
    # Create input text for dependent variable and independent variables. 
    f_inp_txt = inp_txt(df)
    
    # Fit logit model. 
    mdl = logit(f_inp_txt, data=df).fit()
    
    # Create independent variables. 
    X_data = df[df.columns[:-1]]
    
    # Predict y values based on X. 
    y_pred = mdl.predict(X_data)
    
    # Predicted y values are recoded based on threshold value set. 
    y_pred = [1 if i > thold_val else 0 for i in y_pred]
    
    return y_pred 

 
def pro_y_pred_arr(df, thold_val):
    
    '''Create an array of y prediction for each data point for probit model based on threshold value.'''
    
    # Create input text for dependent variable and independent variables. 
    f_inp_txt = inp_txt(df)
    
    # Fit probit model. 
    mdl = probit(f_inp_txt, data=df).fit()
    
    # Create independent variables. 
    X_data = df[df.columns[:-1]]
    
    # Predict y values based on X. 
    y_pred = mdl.predict(X_data)
    
    # Predicted y values are recoded based on threshold value set. 
    y_pred = [1 if i > thold_val else 0 for i in y_pred]
    
    return y_pred 


def roc_curve_plot(df, y_pred_arr, ttl):
    
    '''Plot roc curve.'''
    
    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 
    
    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height 
    
    # Create dependent variable.
    y_data = df[df.columns[-1]]
    
    # Create false positive rate, true positive rate and thresholds values. 
    fpr, tpr, thresholds = roc_curve(y_data, y_pred_arr)
    
    # Plot true positive rate against false positive rate. 
    plt.plot(fpr, tpr, label=ttl)
    
    # Plot line.
    plt.plot([0, 1], [0, 1], 'k--')
    
    # Set x and y labels. 
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    
    # Set title. 
    plt.title('ROC Curve\n(' + str(ttl) + ')', size=20)
    
    # Display plot. 
    plt.show()

# ----------------------------------------------------------------------------------------------

# Subfunction of comb_log_clf_rep_df func.  
def log_clf_rep_df(df, thold_val):
    
    '''Classification report of logit dataframe.'''

    # Create dependent variable.
    y_data = df[df.columns[-1]]

    # Predict y values based on X. 
    y_pred = log_y_pred_arr(df, thold_val) 
    
    # Create clf_report_df dataframe. 
    clf_report_df = pd.DataFrame(classification_report(y_data, y_pred, output_dict=True))
    
    # Create a dataframe based on index.
    df_1 = pd.DataFrame(list(clf_report_df.index))
    
    # Reset clf_report_df dataframe. 
    clf_report_df.reset_index(inplace=True)
    
    # Createa a dataframe based on results of 1 after performing probit.
    df_2 = pd.DataFrame(clf_report_df['1'])
    
    # Create a new dataframe from df_1 and df_2 dataframes.
    df = df_1.join(df_2)
    
    # Add accuracy row into new dataframe. 
    new_row = {0: 'accuracy', '1': clf_report_df['accuracy'][0]}
    df = df.append(new_row, ignore_index=True)
    
    # Add auc score into new dataframe. 
    new_row = {0: 'auc score', '1': roc_auc_score(y_data, y_pred)}
    df = df.append(new_row, ignore_index=True)
    
    return df

def comb_log_clf_rep_df(df_1, df_2, df_3, thold_val):
    
    '''Combine dataframe of logit classification report for train, validate and test data sets.'''
    
    # Dataframe of classification report of logit model of train.
    train_log_clf_rep_df = log_clf_rep_df(df_1, thold_val)
    
    # Dataframe of classification report of logit model of validate.
    val_log_clf_rep_df = log_clf_rep_df(df_2, thold_val)
    
    # Dataframe of classification report of logit model of test.
    test_log_clf_rep_df = log_clf_rep_df(df_3, thold_val)
    
    # Merge the dataframes together. 
    df = train_log_clf_rep_df.merge(val_log_clf_rep_df, how='left', on=0)
    df = df.merge(test_log_clf_rep_df, how='left', on=0)

    # Rename columns of df. 
    columns = {df.columns[0]: 'Logit Model', 
               df.columns[1]: 'Train',
               df.columns[2]: 'Validate',
               df.columns[3]: 'Test'}
    df.rename(columns=columns, inplace=True)

    return df 

# ----------------------------------------------------------------------------------------------

# Subfunction of comb_pro_clf_rep_df func.  
def pro_clf_rep_df(df, thold_val):
    
    '''Classification report of probit dataframe.'''

    # Create dependent variable.
    y_data = df[df.columns[-1]]

    # Predict y values based on X. 
    y_pred = pro_y_pred_arr(df, thold_val) 
    
    # Create clf_report_df dataframe. 
    clf_report_df = pd.DataFrame(classification_report(y_data, y_pred, output_dict=True))
    
    # Create a dataframe based on index.
    df_1 = pd.DataFrame(list(clf_report_df.index))
    
    # Reset clf_report_df dataframe. 
    clf_report_df.reset_index(inplace=True)
    
    # Createa a dataframe based on results of 1 after performing probit.
    df_2 = pd.DataFrame(clf_report_df['1'])
    
    # Create a new dataframe from df_1 and df_2 dataframes.
    df = df_1.join(df_2)
    
    # Add accuracy row into new dataframe. 
    new_row = {0: 'accuracy', '1': clf_report_df['accuracy'][0]}
    df = df.append(new_row, ignore_index=True)
    
    # Add auc score into new dataframe. 
    new_row = {0: 'auc score', '1': roc_auc_score(y_data, y_pred)}
    df = df.append(new_row, ignore_index=True)
    
    return df

def comb_pro_clf_rep_df(df_1, df_2, df_3, thold_val): 
    
    '''Combine dataframe of probit classification report for train, validate and test data sets.'''
    
    # Dataframe of classification report of probit model of train.
    train_pro_clf_rep_df = pro_clf_rep_df(df_1, thold_val)
    
    # Dataframe of classification report of probit model of validate.
    val_pro_clf_rep_df = pro_clf_rep_df(df_2, thold_val)
    
    # Dataframe of classification report of probit model of test.
    test_pro_clf_rep_df = pro_clf_rep_df(df_3, thold_val)
    
    # Merge the dataframes together. 
    df = train_pro_clf_rep_df.merge(val_pro_clf_rep_df, how='left', on=0)
    df = df.merge(test_pro_clf_rep_df, how='left', on=0)

    # Rename columns of df. 
    columns = {df.columns[0]: 'Probit Model', 
               df.columns[1]: 'Train',
               df.columns[2]: 'Validate',
               df.columns[3]: 'Test'}
    df.rename(columns=columns, inplace=True)

    return df

# ----------------------------------------------------------------------------------------------

# Subfunction of linreg_cv_df.
def linreg_reg_r2mean_r2std_msemean_msestd_cv_tup(X, y, ts_siz_1, ts_siz_2, cv):
    
    '''Create a tuple of lists based on linear regression, r2 score mean and std, mse mean and std.'''
    
    reg_lt = [] 
    r2_cv_mean_lt = []
    r2_cv_std_lt = []
    mse_cv_mean_lt = []
    mse_cv_std_lt = [] 
    
    # Linear regression model to run. 
    linreg = LinearRegression(normalize=True)

    # Create reg_text 
    reg_txt = 'Linear Regression'

    # Append polynomial degree into reg_lt. 
    reg_lt.append(reg_txt) 
    
    # Perform train test split. 
    X_train, X_val, X_test, y_train, y_val, y_test = linreg_train_valid_test_split(X, y, ts_siz_1, ts_siz_2) 
    
    # Create pipeline. 
    scaler = MinMaxScaler() 
    linreg = LinearRegression(normalize=True)
    steps = [('scaler', scaler), ('linreg', linreg)]
    pipeline = Pipeline(steps)  
    
    # Fit pipeline. 
    pipeline.fit(X_train, y_train) 
    
    # Peform cross validation for r2 score. 
    r2 = make_scorer(r2_score) 
    r2_cv = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=r2) 

    # Append r2 score mean and std into r2_cv_mean_lt and r2_cv_std_lt. 
    r2_cv_mean_lt.append(r2_cv.mean())
    r2_cv_std_lt.append(r2_cv.std()) 

    # Peform cross validation for mse.  
    mse = make_scorer(mean_squared_error) 
    mse_cv = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=mse)

    # Append mse mean and std into mse_cv_mean_lt and mse_cv_std_lt. 
    mse_cv_mean_lt.append(mse_cv.mean())
    mse_cv_std_lt.append(mse_cv.std()) 
        
    return reg_lt, r2_cv_mean_lt, r2_cv_std_lt, mse_cv_mean_lt, mse_cv_std_lt 

def linreg_cv_df(X, y, ts_siz_1, ts_siz_2, cv): 
    
    '''Create a dataframe of linear regression cross validation results.'''
    
    # Run reg_r2mean_r2std_msemean_msestd_cv_tup func. 
    reg_lt, r2_cv_mean_lt, r2_cv_std_lt, mse_cv_mean_lt, mse_cv_std_lt = linreg_reg_r2mean_r2std_msemean_msestd_cv_tup(X, y, ts_siz_1, ts_siz_2, cv)

    # Create dataframe using r2_cv_mean_lt. 
    df = pd.DataFrame() 
    
    # Create empty columns.
    df[0] = ''
    df[1] = ''
    df[2] = ''
    df[3] = ''

    # Fill up the empty columns using r2_cv_std_lt and deg_lt. 
    df[1] = r2_cv_mean_lt
    df[2] = r2_cv_std_lt
    df[3] = reg_lt 

    # Rename columns. 
    columns = {0: 'params', 
               1: 'mean_train_cv_score', 
               2: 'std_train_cv_score',
               3: 'regression'} 
    df.rename(columns=columns, inplace=True) 

    # Sort df columns based on descending mean_test_score. 
    df = df.sort_values(by=['mean_train_cv_score'], ascending=[False]) 

    return df 

# ----------------------------------------------------------------------------------------------

# Subfunction of poly_reg_r2mean_r2std_msemean_msestd_cv_tup func, poly_ridge_cv_df func, poly_lasso_cv_df func, lasso_alpha_lt func, ridge_alpha_lt func, poly_reg_coef_intcp_rtrain_rval_rtest_msetrain_mseval_msetest_tup func. 
def poly_train_valid_test_split(X, y, deg, ts_siz_1, ts_siz_2):
    
    '''Create train, validation and test data for polynomial data.''' 
    
    # Perform temp-test split. 
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, 
                                                      test_size=ts_siz_1, 
                                                      random_state=0)

    # Convert X_poly and X_hold into polynomial.
    poly = PolynomialFeatures(degree=deg) 
    X_poly = poly.fit_transform(X_temp)  
    X_test = poly.fit_transform(X_test)   

    # Perform train-valid split. 
    X_train, X_val, y_train, y_val = train_test_split(X_poly, y_temp,
                                                      test_size=ts_siz_2, 
                                                      random_state = 0)
    
    return X_train, X_val, X_test, y_train, y_val, y_test 

# Subfunction of poly_cv_combine_df.
def poly_reg_r2mean_r2std_msemean_msestd_cv_tup(X, y, deg, ts_siz_1, ts_siz_2, cv):
    
    '''Create a tuple of lists based on polynomial degree, r2 mean and std, mse mean and std.'''
    
    reg_lt = []
    r2_cv_mean_lt = []
    r2_cv_std_lt = []
    mse_cv_mean_lt = []
    mse_cv_std_lt = [] 
    
    # Linear regression model to run to obtain polynomial regression model. 
    linreg = LinearRegression(normalize=True)
    
    # Create pipeline. 
    scaler = MinMaxScaler() 
    linreg = LinearRegression(normalize=True)
    steps = [('scaler', scaler), ('poly_linreg', linreg)]
    pipeline = Pipeline(steps) 

    for i in range(deg): 

        # Create polynomial degree.  
        deg = i+1 
        reg_txt = 'Polynomial (Deg ' + str(deg) + ')'

        # Append polynomial degree into reg_lt. 
        reg_lt.append(reg_txt) 

        # Perform train test split based on polynomial degree. 
        X_train, X_val, X_test, y_train, y_val, y_test = poly_train_valid_test_split(X, y, deg, ts_siz_1, ts_siz_2) 
        
        # Fit pipeline.  
        pipeline.fit(X_train, y_train)  
        
        # Peform cross validation for r2 score.
        r2 = make_scorer(r2_score)
        r2_cv = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=r2)  

        # Append r2 score mean and std into r2_cv_mean_lt and r2_cv_std_lt. 
        r2_cv_mean_lt.append(r2_cv.mean()) 
        r2_cv_std_lt.append(r2_cv.std()) 

        # Peform cross validation for mse.  
        mse = make_scorer(mean_squared_error) 
        mse_cv = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring=mse)

        # Append mse mean and std into mse_cv_mean_lt and mse_cv_std_lt. 
        mse_cv_mean_lt.append(mse_cv.mean())
        mse_cv_std_lt.append(mse_cv.std())
        
    return reg_lt, r2_cv_mean_lt, r2_cv_std_lt, mse_cv_mean_lt, mse_cv_std_lt 

def poly_cv_combine_df(X, y, deg, ts_siz_1, ts_siz_2, cv): 
    
    '''Create a dataframe of polynomial regression cross validation results.'''
    
    # Run deg_r2mean_r2std_msemean_msestd_cv_tup func. 
    reg_lt, r2_cv_mean_lt, r2_cv_std_lt, mse_cv_mean_lt, mse_cv_std_lt = poly_reg_r2mean_r2std_msemean_msestd_cv_tup(X, y, deg, ts_siz_1, ts_siz_2, cv)

    # Create dataframe using r2_cv_mean_lt. 
    df = pd.DataFrame() 
    
    # Create empty columns. 
    df[0] = ''
    df[1] = ''
    df[2] = ''
    df[3] = ''

    # Fill up the empty columns using r2_cv_std_lt and deg_lt. 
    df[1] = r2_cv_mean_lt
    df[2] = r2_cv_std_lt
    df[3] = reg_lt  

    # Rename columns. 
    columns = {0: 'params', 
               1: 'mean_train_cv_score', 
               2: 'std_train_cv_score', 
               3: 'regression'} 
    df.rename(columns=columns, inplace=True)  

    # Sort df columns based on descending mean_train_cv_score. 
    df = df.sort_values(by=['mean_train_cv_score'], ascending=[False]) 

    return df 

# ----------------------------------------------------------------------------------------------

def ridge_alpha_lt(X, y, deg, ts_siz_1, ts_siz_2, cv):
    
    '''Create a list of optimized alpha for ridge regression.'''
    
    alpha_lt = [] 
    
    # Set ridgecv model.
    rcv = RidgeCV(normalize=True, cv=cv) 
    
    # Create pipeline. 
    scaler = MinMaxScaler() 
    steps = [('scaler', scaler), ('ridge_cv', rcv)]
    pipeline = Pipeline(steps)  
    
    for i in range(deg): 

        # Create deg. 
        deg = i+1

        # Create train, validate and test datasets.
        X_train, X_val, X_test, y_train, y_val, y_test = poly_train_valid_test_split(X, y, deg, ts_siz_1, ts_siz_2)

        # Fit pipeline. 
        pipeline.fit(X_train, np.ravel(y_train))  

        # Append optimized alpha into alpha_lt. 
        alpha_lt.append(rcv.alpha_)
    
    return alpha_lt 


def poly_ridge_cv_df(X, y, deg, ts_siz_1, ts_siz_2, params, cv):
    
    '''Create dataframe of grid search cross validation results on polynomial and ridge regression.'''
    
    # Perform train test split. 
    X_train, X_val, X_test, y_train, y_val, y_test = poly_train_valid_test_split(X, y, deg, ts_siz_1, ts_siz_2) 
    
    # Create pipeline. 
    scaler = MinMaxScaler() 
    ridge = Ridge(normalize=True, max_iter = 10000000, random_state=0)
    steps = [('scaler', scaler), ('ridge', ridge)]
    pipeline = Pipeline(steps)   
    
    # Fit pipeline. 
    pipeline.fit(X_train, y_train)
    
    # Parameters to be used for gridsearch. 
    parameters = params 

    # Perform gridsearch. 
    clf = GridSearchCV(pipeline, param_grid = parameters, cv=cv)  
    clf.fit(X_train, y_train)   

    # Create dataframe for gridsearch results. 
    cv_results_df = pd.DataFrame(clf.cv_results_) 

    # Sort gridsearch results by column 'rank_test_score'. 
    cv_results_df.sort_values(by='rank_test_score', ascending=True)
    
    return cv_results_df


def lasso_alpha_lt(X, y, deg, ts_siz_1, ts_siz_2, cv):
    
    '''Create a list of optimized alpha for lasso regression.'''
    
    alpha_lt = [] 
    
    # Set lassocv model.
    lcv = LassoCV(max_iter = 10000000, normalize=True, random_state=0)
    
    # Create pipeline. 
    scaler = MinMaxScaler() 
    steps = [('scaler', scaler), ('lasso_cv', lcv)]
    pipeline = Pipeline(steps) 
    
    for i in range(deg): 

        # Create deg. 
        deg = i+1

        # Create train, validate and test datasets. 
        X_train, X_val, X_test, y_train, y_val, y_test = poly_train_valid_test_split(X, y, deg, ts_siz_1, ts_siz_2)
        
        # Fit pipeline. 
        pipeline.fit(X_train, np.ravel(y_train))
        
        # Append optimized alpha into alpha_lt. 
        alpha_lt.append(lcv.alpha_)
    
    return alpha_lt


def poly_lasso_cv_df(X, y, deg, ts_siz_1, ts_siz_2, params, cv):
    
    '''Create dataframe of grid search cross validation results on polynomial and lasso regression.'''
    
    # Perform train test split. 
    X_train, X_val, X_test, y_train, y_val, y_test = poly_train_valid_test_split(X, y, deg, ts_siz_1, ts_siz_2)
        
    # Create pipeline. 
    scaler = MinMaxScaler() 
    lasso = Lasso(normalize=True, max_iter = 10000000, random_state=0)
    steps = [('scaler', scaler), ('lasso', lasso)]
    pipeline = Pipeline(steps)   
    
    # Fit pipeline. 
    pipeline.fit(X_train, y_train) 
    
    # Parameters to be used for gridsearch. 
    parameters = params 

    # Perform gridsearch. 
    clf = GridSearchCV(pipeline, param_grid = parameters, cv=cv) 
    clf.fit(X_train, y_train)  
    
    # Create dataframe for gridsearch results. 
    cv_results_df = pd.DataFrame(clf.cv_results_)
    
    # Sort gridsearch results by column 'rank_test_score'. 
    cv_results_df.sort_values(by='rank_test_score', ascending=True)
    
    return cv_results_df

# ----------------------------------------------------------------------------------------------

# Subfunction of linreg_train_val_test_df func.   
def linreg_reg_coef_intcp_rtrain_rval_rtest_cftrain_cfval_cftest_tup(X, y, ts_siz_1, ts_siz_2):
    
    '''Create a tuple of lists of degree, cofficient, intercept, r and cost function for train, validate and test for linear regression.''' 

    # Perform train-valid-test split.   
    X_train, X_val, X_test, y_train, y_val, y_test = linreg_train_valid_test_split(X, y, ts_siz_1, ts_siz_2) 
    
    reg_lt = [] 
    coef_lt = [] 
    intcp_lt = [] 
    r_train_lt = [] 
    r_val_lt = []  
    r_test_lt = [] 
    cf_train_lt = [] 
    cf_val_lt = [] 
    cf_test_lt = []  
    
    # Create pipeline. 
    scaler = MinMaxScaler() 
    linreg = LinearRegression(normalize=True) 
    steps = [('scaler', scaler), ('linreg', linreg)]
    pipeline = Pipeline(steps)  
    
    # Fit pipeline. 
    pipeline.fit(X_train, y_train)
    
    # Append results for polynomial regression. 
    reg_lt.append((None, 'Linear', None))
    coef_lt.append(linreg.coef_[0]) 
    intcp_lt.append(linreg.intercept_[0]) 
    
    # Calculate mse.  
    y_pred_train = pipeline.predict(X_train) 
    y_pred_val = pipeline.predict(X_val) 
    y_pred_test = pipeline.predict(X_test)   
    mse_train = mean_squared_error(y_train, y_pred_train) 
    mse_val = mean_squared_error(y_val, y_pred_val)
    mse_test = mean_squared_error(y_test, y_pred_test)   
    
    # Append results for polynomial regression. 
    r_train_lt.append(np.sqrt(pipeline.score(X_train, y_train))) 
    r_val_lt.append(np.sqrt(pipeline.score(X_val, y_val)))
    r_test_lt.append(np.sqrt(pipeline.score(X_test, y_test))) 
    cf_train_lt.append(mse_train/(2*len(X_train))) 
    cf_val_lt.append(mse_val/(2*len(X_val))) 
    cf_test_lt.append(mse_test/(2*len(X_test)))   
    
    return reg_lt, coef_lt, intcp_lt, r_train_lt, r_val_lt, r_test_lt, cf_train_lt, cf_val_lt, cf_test_lt 

def linreg_train_val_test_df(X, y, ts_siz_1, ts_siz_2):  
    
    '''Create a dataframe with columns: degree, regression, alpha, model coefficients, model intercept, train score, validate score, test score, train cost function, validate cost function, test cost function for linear regression.'''
    
    # Create lists of degree, cofficient, intercept and correlation and mse of train, validate and test.  
    reg_lt, coef_lt, intcp_lt, r_train_lt, r_val_lt, r_test_lt, cf_train_lt, cf_val_lt, cf_test_lt = linreg_reg_coef_intcp_rtrain_rval_rtest_cftrain_cfval_cftest_tup(X, y, ts_siz_1, ts_siz_2) 
    
    # Create a dataframe.
    df = pd.DataFrame(reg_lt)  
    
    # Create empty columns. 
    df[3] = ''
    df[4] = ''
    df[5] = '' 
    df[6] = ''
    df[7] = ''
    df[8] = ''
    df[9] = ''
    df[10] = ''
    
    # Input values into columns.
    df[3] = coef_lt
    df[4] = intcp_lt
    df[5] = r_train_lt
    df[6] = r_val_lt 
    df[7] = r_test_lt 
    df[8] = cf_train_lt 
    df[9] = cf_val_lt 
    df[10] = cf_test_lt 
    
    # Rename columns. 
    columns = {0: 'Degree', 
               1: 'Regression',
               2: 'alpha',
               3: 'Model Coefficients',
               4: 'Model Intercept',
               5: 'Train Correlation',
               6: 'Validate Correlation', 
               7: 'Test Correlation', 
               8: 'Train Cost Function', 
               9: 'Validate Cost Function', 
               10: 'Test Cost Function'} 
    df.rename(columns=columns, inplace=True)  
    
    # # Sort columns based on ascending 'Validate Cost Function', descending 'Validate Correlation', ascending 'Test Cost Function', descending 'Test Correlation', ascending 'Train Cost Function' and descending 'Train Correlation'.  
    # df = df.sort_values(by=['Validate Cost Function', 'Validate Correlation', 'Test Cost Function', 'Test Correlation', 'Train Cost Function', 'Train Correlation'], ascending=[True, False, True, False, True, False])
    
    # Sort columns based on ascending 'Validate Cost Function', 'Test Cost Function', and 'Train Cost Function' and descending 'Validate Correlation', 'Test Correlation', and 'Train Correlation'. 
    df = df.sort_values(by=['Validate Cost Function', 'Test Cost Function', 'Train Cost Function',
                            'Validate Correlation', 'Test Correlation', 'Train Correlation'], 
                        ascending=[True, True, True, False, False, False])
    
    return df 

# ----------------------------------------------------------------------------------------------

# Subfunction of poly_reg_coef_intcp_rtrain_rval_rtest_msetrain_mseval_msetest_tup func. 
def mse_train_val_test_tup(mdl, X_train, X_val, X_test, y_train, y_val, y_test):
    
    '''Create a tuple with mse train, mse val, mse test.'''
    
    # Calculate y predict.
    y_pred_train = mdl.predict(X_train) 
    y_pred_val = mdl.predict(X_val) 
    y_pred_test = mdl.predict(X_test)  
    
    # Calculate mse. 
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_val = mean_squared_error(y_val, y_pred_val)
    mse_test = mean_squared_error(y_test, y_pred_test) 
    
    return mse_train, mse_val, mse_test 

# Subfunction of poly_ridge_lasso_train_val_test_df func.   
def poly_reg_coef_intcp_rtrain_rval_rtest_cftrain_cfval_cftest_tup(X, y, deg, ts_siz_1, ts_siz_2, ridge_alpha, lasso_alpha):
    
    '''Create a tuple of lists of degree, cofficient, intercept, accuracy score and cost function for train, validate and hold for polynomial regression.''' 

    # Perform train-valid-test split.    
    X_train, X_val, X_test, y_train, y_val, y_test = poly_train_valid_test_split(X, y, deg, ts_siz_1, ts_siz_2) 
    
    reg_lt = [] 
    coef_lt = [] 
    intcp_lt = [] 
    r_train_lt = [] 
    r_val_lt = [] 
    r_test_lt = [] 
    cf_train_lt = []
    cf_val_lt = [] 
    cf_test_lt = [] 
    
    # Polynomial regression # 
    # Create pipeline for polynomial regression.
    scaler = MinMaxScaler() 
    poly = LinearRegression(normalize=True) 
    steps = [('scaler', scaler), ('poly', poly)]
    pipeline_poly = Pipeline(steps)  
    
    # Fit pipeline for polynomial regression. 
    pipeline_poly.fit(X_train, y_train)  
    
    # Calculate mse for polynomial regression. 
    mse_train, mse_val, mse_test = mse_train_val_test_tup(pipeline_poly, X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Append results for polynomial regression. 
    reg_lt.append((deg, 'Polynomial', None))
    coef_lt.append(poly.coef_[0]) 
    intcp_lt.append(poly.intercept_[0])
    r_train_lt.append(np.sqrt(abs(pipeline_poly.score(X_train, y_train))))
    r_val_lt.append(np.sqrt(abs(pipeline_poly.score(X_val, y_val))))
    r_test_lt.append(np.sqrt(abs(pipeline_poly.score(X_test, y_test))))    
    cf_train_lt.append(mse_train/(2*len(X_train))) 
    cf_val_lt.append(mse_val/(2*len(X_val))) 
    cf_test_lt.append(mse_test/(2*len(X_test)))   
    
    # Polynomial and ridge regression #   
    # Create pipeline for polynomial and ridge regression.
    scaler = MinMaxScaler() 
    ridge = Ridge(alpha=ridge_alpha, solver='saga', max_iter = 10000000, normalize=True, random_state=0)
    steps = [('scaler', scaler), ('poly_ridge', ridge)]
    pipeline_ridge = Pipeline(steps)  
    
    # Fit pipeline for polynomial and ridge regression. 
    pipeline_ridge.fit(X_train, y_train)  
    
    # Calculate mse for polynomial regression with ridge regression. 
    mse_train, mse_val, mse_test = mse_train_val_test_tup(pipeline_ridge, X_train, X_val, X_test, y_train, y_val, y_test)

    # Append results for polynomial with ridge regression. 
    reg_lt.append((deg, 'Polynomial & Ridge', ridge_alpha))
    coef_lt.append(ridge.coef_[0]) 
    intcp_lt.append(ridge.intercept_[0])
    r_train_lt.append(np.sqrt(abs(pipeline_ridge.score(X_train, y_train))))
    r_val_lt.append(np.sqrt(abs(pipeline_ridge.score(X_val, y_val))))
    r_test_lt.append(np.sqrt(abs(pipeline_ridge.score(X_test, y_test))))    
    cf_train_lt.append(mse_train/(2*len(X_train))) 
    cf_val_lt.append(mse_val/(2*len(X_val))) 
    cf_test_lt.append(mse_test/(2*len(X_test)))   
    
    # Polynomial and lasso regression # 
    # Create pipeline for polynomial and lasso regression.
    scaler = MinMaxScaler() 
    lasso = Lasso(alpha=lasso_alpha, max_iter = 10000000, normalize=True, random_state=0) 
    steps = [('scaler', scaler), ('poly_lasso', lasso)]
    pipeline_lasso = Pipeline(steps)  
    
    # Fit pipeline for polynomial and lasso regression. 
    pipeline_lasso.fit(X_train, y_train) 

    # Calculate mse. 
    mse_train, mse_val, mse_test = mse_train_val_test_tup(pipeline_lasso, X_train, X_val, X_test, y_train, y_val, y_test)

    # Append results for polynomial with lasso regression. 
    reg_lt.append((deg, 'Polynomial & Lasso', lasso_alpha)) 
    coef_lt.append(lasso.coef_)
    intcp_lt.append(lasso.intercept_[0]) 
    r_train_lt.append(np.sqrt(abs(pipeline_lasso.score(X_train, y_train))))
    r_val_lt.append(np.sqrt(abs(pipeline_lasso.score(X_val, y_val)))) 
    r_test_lt.append(np.sqrt(abs(pipeline_lasso.score(X_test, y_test))))   
    cf_train_lt.append(mse_train/(2*len(X_train))) 
    cf_val_lt.append(mse_val/(2*len(X_val))) 
    cf_test_lt.append(mse_test/(2*len(X_test)))   

    return reg_lt, coef_lt, intcp_lt, r_train_lt, r_val_lt, r_test_lt, cf_train_lt, cf_val_lt, cf_test_lt 

def poly_ridge_lasso_train_val_test_df(X, y, deg, ts_siz_1, ts_siz_2, ridge_alpha, lasso_alpha):  
    
    '''Create a dataframe with columns: degree, regression, alpha, model coefficients, model intercept, train corr, validate corr, test corr, train cost function, validate cost function, test cost function for polynomial regression.'''
    
    # Create lists of degree, coefficient, intercept, and train, validate and test of correlation and cost function. 
    reg_lt, coef_lt, intcp_lt, r_train_lt, r_val_lt, r_test_lt, cf_train_lt, cf_val_lt, cf_test_lt  = poly_reg_coef_intcp_rtrain_rval_rtest_cftrain_cfval_cftest_tup(X, y, deg, ts_siz_1, ts_siz_2, ridge_alpha, lasso_alpha) 
    
    # Create a dataframe.
    df = pd.DataFrame(reg_lt)  
    
    # Create empty columns. 
    df[3] = ''
    df[4] = ''
    df[5] = ''
    df[6] = ''
    df[7] = ''
    df[8] = '' 
    df[9] = ''
    df[10] = ''
    
    # Input values into columns.
    df[3] = coef_lt
    df[4] = intcp_lt
    df[5] = r_train_lt
    df[6] = r_val_lt 
    df[7] = r_test_lt 
    df[8] = cf_train_lt # formerly mse 
    df[9] = cf_val_lt 
    df[10] = cf_test_lt  
    
    # Rename columns. 
    columns = {0: 'Degree', 
               1: 'Regression',
               2: 'alpha',
               3: 'Model Coefficients',
               4: 'Model Intercept',
               5: 'Train Correlation',
               6: 'Validate Correlation', 
               7: 'Test Correlation',
               8: 'Train Cost Function', 
               9: 'Validate Cost Function', 
               10: 'Test Cost Function'} 
    df.rename(columns=columns, inplace=True)   
    
    # Sort columns based on ascending 'Validate Cost Function', 'Test Cost Function', and 'Train Cost Function' and descending 'Validate Correlation', 'Test Correlation', and 'Train Correlation'. 
    df = df.sort_values(by=['Validate Cost Function', 'Test Cost Function', 'Train Cost Function',
                            'Validate Correlation', 'Test Correlation', 'Train Correlation'], 
                        ascending=[True, True, True, False, False, False])
    
    return df

# ----------------------------------------------------------------------------------------------

def cfplot(df, col_1, col_2, col_3, opt_deg):
    
    '''Plot cost functions for train, validate and test for various polynomial regressions.'''

    # Set graph layout. 
    myfcp_gra.set_sns_font_dpi()
    myfcp_gra.set_sns_large_white() 

    # Modify figure size.
    plt.figure(figsize=(15,6)) # width x height
    
    # Plot cost functions. 
    sns.lineplot(x='Degree', y=col_1, data=df, label=col_1)
    sns.lineplot(x='Degree', y=col_2, data=df, label=col_2)
    sns.lineplot(x='Degree', y=col_3, data=df, label=col_3)
    
    # Vertical line to identify the optimized degree of polynomial. 
    plt.axvline(x = opt_deg, color = 'black')
    
    # Set x and y labels. 
    plt.xlabel('Degree of Polynomial')
    plt.ylabel('Cost Function')

    # Set title. 
    plt.title('Cost Function\n(Train, Validate, Test)', size=20)

    # Display plot. 
    plt.show() 

