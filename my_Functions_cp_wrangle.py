#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  8 18:20:40 2022

@author: nelsonng
"""

import numpy as np 

from datetime import date 

# ----------------------------------------------------------------------------------------------

def add_diff_days_col(df, col, new_col, curr_date): 
    
    '''Add column for the amount of days from the current date.'''
    
    # Create column. 
    df[new_col] = ''

    for i, j in enumerate(df[col]):

        # Convert j into str format.
        txt = str(j)

        # Convert to year, month and day in int format. 
        yr = int(txt[:4])
        mth = int(txt[5:7])
        day = int(txt[8:10])

        # Calculate the difference in number days. 
        delta = curr_date - date(yr, mth, day)

        # Input the number of days difference into the created column. 
        df[new_col].iloc[i] = delta.days    

# ----------------------------------------------------------------------------------------------

# Subfunction of bins_lt func and labels_lt func.   
def bins_tp_lt(df, col, n):
    
    '''Create a bins list filled with tuples.'''
    
    bins_tp_lt = [] 
    
    # Iterate to create the bins. 
    for i, j in enumerate(range(int(np.max(df[col])//n + 1))):
        
        # Segregate based on i.
        if i == 0:
            
            # Append index and -np.inf.
            bins_tp_lt.append((i, -np.inf))

        else: 
            
            # Append index and increment of n.
            bins_tp_lt.append((i, i*n))
            
    return bins_tp_lt 

def bins_lt(df, col, n):
    
    '''Create a list of bins.''' 
    
    # Create a list of bins in tuples.
    f_bins_tp_lt = bins_tp_lt(df, col, n) 
    
    # Create a list of bins. 
    bins_lt = [i[1] for i in f_bins_tp_lt] 
    
    # Replace the last index with np.inf. 
    bins_lt[-1] = np.inf

    return bins_lt 


def label_lt(df, col, n): 
    
    '''Create a list of labels.'''
    
    label_lt = []
    num_lt = [] 
    
    # Create a list of bins in tuples.
    f_bins_tp_lt = bins_tp_lt(df, col, n) 

    for i in f_bins_tp_lt:
        
        # First index of f_bins_tp_lt. 
        if i[0] == f_bins_tp_lt[0][0]:
            continue
        
        # Second index of f_bins_tp_lt. 
        if i[0] == f_bins_tp_lt[1][0]:
            txt = '< ' + str(i[1])
            label_lt.append(txt)
            num_lt.append(i[1])
        
        # Last index of f_bins_tp_lt. 
        elif i[0] == f_bins_tp_lt[-1][0]:
            txt = '> ' + str(num_lt[-1])  
            label_lt.append(txt)
            num_lt.append(i[1])
        
        # Middle indexs of f_bins_tp_lt. 
        else:
            txt = str(num_lt[-1]) + ' - ' + str(i[1])
            label_lt.append(txt)
            num_lt.append(i[1]) 
            
    return label_lt

# ----------------------------------------------------------------------------------------------

