#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 14:27:29 2022

@author: nelsonng
"""

import numpy as np 

from scipy import stats 

# ----------------------------------------------------------------------------------------------

def print_null_col(df): 
    
    '''Print null values of columns.'''
    
    # Null values per column
    print('\033[1m' + 'Column              No. of Null Values' + '\033[0m') 
    print(df.isnull().sum())
    

def print_50_pct_null_col(df, copy_df):
    
    '''Print columns that have >50 precent of null values.'''
    
    # Create deep copy of df and df_copy.
    df = df.copy(deep=True)
    copy_df = copy_df.copy(deep=True)

    # Identify columns that have null values. 
    column_has_nan = df.columns[df.isnull().any()]

    # Search and drop columns that have at least 50% of null data.
    for column in column_has_nan:

        # Ensure that there are at least 50% of null data.
        if df[column].isnull().sum()/df.shape[0] > 0.50:
            df.drop(column, 1, inplace=True) 

    # Identify columns having >50% of null values. 
    original_columns = [i for i in copy_df.columns]
    new_columns = [i for i in df.columns]
    removed_columns = [i for i in original_columns if i not in new_columns]
    
    # Print columns having >50% of null values. 
    print('\033[1m' + 'Columns having >50% of Null Values:' + '\033[0m')
    for i in removed_columns:
        print(i)


def remove_null_row(df, col):
    
    '''Remove null row of dataframe.'''
    
    # Identify the index of the null values.
    labels=[x for x in df[df[col].isnull()].index]

    # Drop null rows of df basing on identified index.
    df.drop(labels=labels, axis=0, inplace=True)


def duplicates_df(df, col_lt):
    
    '''Create a dataframe for duplicates.'''
    
    # Create a boolean values to identify duplicate rows.
    duplicates = df.duplicated(subset=col_lt, keep=False)
    
    # Create a boolean values to identify duplicate rows. 
    df = df[duplicates]
    
    return df


def print_less_zero_cols(df):
    
    '''Print columns of int64 and float64 datatypes with value less than 0.'''
    
    col_ind_lt = []
    
    # Append the sorted index based on datatypes to col_ind_lt.
    for i, j in enumerate(df.dtypes):
        if j == np.int64 or j == np.float64:
            col_ind_lt.append(i) 
    
    # Print columns that have numeric value below 0.  
    print('\033[1m' + 'Columns of int64 and float64 datatypes with value less than 0:' + '\033[0m')
    
    # Iterate over the index of the dataframe column. 
    for i in col_ind_lt:
        
        # Iterate over the selected dataframe column. 
        for j in df[df.columns[i]]:
            
            # Print column if numeric value is below 0.  
            if j < 0:
                txt = i
                print(txt)
                

def outlier_df(df, col, thold): 

    '''Create a dataframe for outliers.'''
    
    # Find the z score for each data point based on column.
    z = np.abs(stats.zscore(df[col]))
    
    # Create dataframe for outliers. 
    outlier_df = df[z > thold]
    
    return outlier_df 


def inlier_df(df, col, thold): 

    '''Create a dataframe for inliers.'''
    
    # Find the z score for each data point based on column.
    z = np.abs(stats.zscore(df[col]))
    
    # Create dataframe for inliers. 
    inlier_df = df[z < thold]
    
    return inlier_df


def print_outliers(df, copy_df):
    
    '''Print number and percentage of removed outlier rows.'''
    
    # Calculate number and percentage of removed outlier rows.  
    row_rmv = copy_df.shape[0] - df.shape[0]
    pct_rmv = row_rmv/copy_df.shape[0]*100
    
    # Set text for row 1. 
    txt_1 = "Number of outlier rows removed: "
    row_rmv_txt = str(row_rmv)
    
    # Set text for row 2. 
    txt_2 = 'Percentage of outlier rows removed: ' 
    pct_rmv_txt = str(round(pct_rmv, 2)) + '%' 
    
    # Print text. 
    print('\033[1m' + 'Outlier Rows' + '\033[0m')
    print(txt_1 + row_rmv_txt)
    print(txt_2 + pct_rmv_txt) 

