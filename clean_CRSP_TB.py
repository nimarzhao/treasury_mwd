#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: File to clean CRSP data
@author: nimarzhao
@email: nimar.zhao@maths.ox.ac.uk

"""

# IMPORTS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

""" TO DO:

    0. Install libraries: # pip install fredapi pandas numpy matplotlib
    1. Set directory (optional)
    2. Load CRSP Treasury data, labelled as 'crsp_raw.csv'

"""

# TODO 1. Set the working directory
# os.chdir()
print(f"Current working directory: {os.getcwd()}")  # verify the current working directory

# TODO 2. Load the CSV file into a DataFrame (may take a while)
crsp_raw = pd.read_csv('data/crsp_raw.csv')



'''
    1. Clean by dropping duplicates
'''

# Check the first few rows to verify that the file has been loaded correctly
print(crsp_raw.head())

# Sort the DataFrame so that rows with non-NaN TDTOTOUT come first
crsp_raw_sorted = crsp_raw.sort_values(by='TDTOTOUT', ascending=False)

# Drop duplicates based on CRSPID, keeping the first occurrence (which now has the non-NaN TDTOTOUT)
crsp_dropped = crsp_raw_sorted.drop_duplicates(subset='CRSPID', keep='first')


'''
    2. Fix public holdings
'''

# Group by CRSPID and compute the average of TDPUBOUT, ignoring NaN values
average_TDPUBOUT_per_CRSPID = crsp_raw.groupby('CRSPID')['TDPUBOUT'].mean()

# Fill the NaN values in average_TDPUBOUT_per_CRSPID with the corresponding TDTOTOUT values
# For each CRSPID where TDPUBOUT is NaN, we will replace it with the corresponding TDTOTOUT
average_TDPUBOUT_per_CRSPID_filled = average_TDPUBOUT_per_CRSPID.copy()

# Update the NaN values with the TDTOTOUT values for the respective CRSPIDs
nan_crspids = average_TDPUBOUT_per_CRSPID_filled[average_TDPUBOUT_per_CRSPID_filled.isna()].index

# Replace NaNs with the average of TDTOTOUT for those CRSPIDs
replacement_values = crsp_raw.groupby('CRSPID')['TDTOTOUT'].mean()
average_TDPUBOUT_per_CRSPID_filled.update(replacement_values)

crsp_dropped['TDPUBOUT'] = crsp_dropped['CRSPID'].map(average_TDPUBOUT_per_CRSPID_filled)


'''
    3. Remove those with maturity before 1990.
'''

# Converting the TMATDT column to datetime to perform filtering based on date
crsp_dropped['TMATDT'] = pd.to_datetime(crsp_dropped['TMATDT'], errors='coerce')

# Defining the cutoff date
cutoff_date = pd.Timestamp('1990-01-01')

# Filtering out rows with maturity date after January 1990
crsp_clean = crsp_dropped[crsp_dropped['TMATDT'] >= cutoff_date]

# Drop rows where TDTOTOUT is missing (NaN)
crsp_clean = crsp_clean.dropna(subset=['TDTOTOUT'])


'''
    3. Export
'''

# Convert all date columns to first date of the respective month.
# Identifying the columns that contain date-like values.
date_columns = ['TDATDT', 'TMATDT']

# Convert these columns to datetime and set them to the first day of the month
for col in date_columns:
    crsp_clean[col] = pd.to_datetime(crsp_clean[col]).dt.to_period('M').dt.to_timestamp('M')

# Export
crsp_clean.to_csv("data/cleaned/crsp_clean.csv")





