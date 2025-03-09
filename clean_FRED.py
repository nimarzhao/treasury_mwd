#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@file: File to clean FRED data
@author: nimarzhao
@email: nimar.zhao@maths.ox.ac.uk

"""

# IMPORTS
from fredapi import Fred
import pandas as pd
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import os
import statsmodels.api as sm

""" TO DO:

    0. Install libraries: # pip install fredapi pandas numpy matplotlib
    1. Set directory (optional)
    2. Set FRED API Key for GDP data

"""

# TODO 1. Set the working directory
#os.chdir()
print(f"Current working directory: {os.getcwd()}")  # verify the current working directory

# TODO 2. Initialize FRED with your API key
fred = Fred(api_key='### YOUR KEY HERE ###')



'''
    GDP
'''

# Fetch and preprocess GDP data
gdp = fred.get_series('GDP')
gdp = pd.DataFrame(gdp, columns=['GDP'])
gdp['GDP'] *= 1000  # Convert to correct units
gdp.index = pd.to_datetime(gdp.index)

# Restrict data to the period between Jan 1990 and Dec 2023
gdp = gdp.loc['1990-01-01':'2024-12-31']

# Compute log GDP and resample to monthly frequency
log_gdp = np.log(gdp).resample('M').mean()
log_gdp_ipolate = log_gdp.interpolate(method='linear')
gdp_ipolate = np.exp(log_gdp_ipolate)

# Apply the Hodrick-Prescott filter
hp_gap, hp_trend = sm.tsa.filters.hpfilter(log_gdp_ipolate, lamb=129600)
# lamb=129600 is the smoothing parameter for monthly data 
# (adjust this based on the frequency of your data).


'''
    Inflation
'''
 
# Fetch and compute inflation (YoY change in CPI)
cpi = fred.get_series('CPIAUCNS')
inf_ttm = cpi.pct_change(12)

# Restrict data to December 2023
cpi = cpi.loc[:'2023-12-01']
inf_ttm = inf_ttm.loc[:'2023-12-01']

# Extend inflation series for 2024-2070 using Fed median projections
future_dates = pd.date_range(start='2024-01-01', end='2070-12-31', freq='M')
inf_extension = pd.Series([0.026] * 12 + [0.022] * 12 + [0.020] * (len(future_dates) - 24), index=future_dates)
inf_ttm = pd.concat([inf_ttm, inf_extension])

# Project CPI values using the last known value (Dec 2023)
last_cpi_value = cpi.iloc[-1]
cpi_projection = [(last_cpi_value := last_cpi_value * (1 + rate / 12)) for rate in inf_extension]
cpi_extension = pd.Series(cpi_projection, index=future_dates)
cpi = pd.concat([cpi, cpi_extension])


'''
    More FRED data
'''

three_month = fred.get_series('DTB3').resample('M').mean() / 100
michigan = fred.get_series('MICH')
unemployment_rate = fred.get_series('UNRATE') / 100
vix = fred.get_series('VIXCLS').resample('M').mean()

# Load Cleveland Fed Inflation Expectations Data
# inf_rp = pd.read_csv('data/cleveland_fed_inflationexpectations_tenyearexpectedinflation.csv', index_col=0, parse_dates=True)


'''
    Export
'''

# Align all indices to the end of the month
def align_to_month_end(df):
    df.index += pd.offsets.MonthEnd(0)
    return df

# List of dataframes to align
dataframes = [gdp_ipolate, log_gdp_ipolate, cpi, inf_ttm, three_month, unemployment_rate, vix, michigan]
dataframes = [align_to_month_end(df) for df in dataframes]
gdp_ipolate, log_gdp_ipolate, cpi, inf_ttm, three_month, unemployment_rate, vix, michigan = dataframes
# dataframes = [gdp_ipolate, log_gdp_ipolate, cpi, inf_ttm, three_month, unemployment_rate, vix, michigan, inf_rp]
# dataframes = [align_to_month_end(df) for df in dataframes]
# gdp_ipolate, log_gdp_ipolate, cpi, inf_ttm, three_month, unemployment_rate, vix, michigan, inf_rp = dataframes


# Combine all series into a single DataFrame
fred_data = pd.concat([
    gdp_ipolate, log_gdp_ipolate, cpi, inf_ttm, three_month,
    unemployment_rate, vix, michigan, hp_gap, hp_trend
], axis=1)
# fred_data = pd.concat([
#     gdp_ipolate, log_gdp_ipolate, cpi, inf_ttm, three_month,
#     unemployment_rate, vix, michigan, hp_gap, hp_trend, inf_rp
# ], axis=1)


# Rename columns for clarity
fred_data.columns = [
    'GDP', 'Log_GDP', 'CPI', 'Inflation', 'DTB3', 'UNRATE', 'VIX',
    'Michigan', 'HP_Gap', 'HP_Trend'
]
# fred_data.columns = [
#     'GDP', 'Log_GDP', 'CPI', 'Inflation', 'DTB3', 'UNRATE', 'VIX',
#     'Michigan', 'HP_Gap', 'HP_Trend', 'Cleveland_Exp_Inflation',
#     'Cleveland_Real_RP', 'Cleveland_Inf_RP'
# ]
# Export the combined DataFrame to a CSV file
fred_data.to_csv('data/fred_data.csv', index=True)



