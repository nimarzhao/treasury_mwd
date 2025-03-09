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

""" VERIFY:

    0. (Assumed) install libraries: # pip install fredapi pandas numpy matplotlib
    1. Run clean_FRED and clean_CRSP_TB to obtain:
        - fred_data.csv
        - crsp_clean.csv

"""


'''
    Setup: Load data and description
'''

# 1. Load FRED data
fred_data = pd.read_csv('data/fred_data.csv',index_col = 0)
fred_data.index = pd.to_datetime(fred_data.index)

# 2. Load CRSP data
crsp_clean = pd.read_csv('data/crsp_clean.csv')

# Datetime conversion
crsp_clean['TDATDT'] = pd.to_datetime(crsp_clean['TDATDT'])
crsp_clean['TFCPDT'] = pd.to_datetime(crsp_clean['TFCPDT'])
crsp_clean['TMATDT'] = pd.to_datetime(crsp_clean['TMATDT'])

# Split data for ITYPE (IWHY) equal to 11 or 12 (TIPS) or not (Nominal)
crsp_nominal = crsp_clean[~crsp_clean['ITYPE'].isin([11, 12])]
crsp_tips = crsp_clean[(crsp_clean['ITYPE'] == 11) | (crsp_clean['ITYPE'] == 12)]

# Generate monthly dates from January 1990 to December 2070
monthly_dates = pd.date_range(start='1990-01-01', end='2070-12-31', freq='M')




'''
    Nominal Payment Schedule
'''

# Function to compute the coupon payments schedule for each bond
def compute_payments_for_bond(row):
    bond_id = row['CRSPID']
    first_coupon_date = pd.to_datetime(row['TFCPDT']) if not pd.isna(row['TFCPDT']) else pd.to_datetime(row['TMATDT'])
    maturity_date = pd.to_datetime(row['TMATDT'])
    coupon_rate = row['TCOUPRT'] / 100  # Convert percentage to decimal
    bond_value = row['TDTOTOUT']
    
    # Semi-annual coupon payment
    coupon_payment = bond_value * (coupon_rate / 2)
    
    # Initialize a series for this bond's payment schedule
    bond_payments = pd.Series(0, index=monthly_dates)
    
    # Make semi-annual coupon payments starting from first coupon date until maturity
    coupon_dates = pd.date_range(start=first_coupon_date, end=maturity_date, freq='6M')
    for date in coupon_dates:
        bond_payments[date] = coupon_payment
        
    bond_payments[maturity_date] += bond_value  # Add both the final coupon and the bond value
    
    return bond_payments


# Initialize an empty DataFrame for the payment schedule
nominal_schedule = pd.DataFrame(index=monthly_dates)

# Apply the function to each bond in the cleaned dataset and store the result in the payment schedule
for _, bond_row in crsp_nominal.iterrows():
    bond_id = bond_row['CRSPID']
    nominal_schedule[bond_id] = compute_payments_for_bond(bond_row)




'''
    TIPS Payment Schedule
'''

# Function to compute the coupon payments schedule for each inflation-linked bond
def compute_payments_for_inflation_bond(row, inflation_index):
    bond_id = row['CRSPID']
    first_coupon_date = (pd.to_datetime(row['TFCPDT']) if not pd.isna(row['TFCPDT']) else pd.to_datetime(row['TMATDT'])) + pd.offsets.MonthEnd(0)
    maturity_date = pd.to_datetime(row['TMATDT']) + pd.offsets.MonthEnd(0)
    coupon_rate = row['TCOUPRT'] / 100  # Convert percentage to decimal
    bond_value = row['TDTOTOUT']  # Bond value at issuance
    
    # Initialize a series for this bond's payment schedule
    bond_payments = pd.Series(0, index=monthly_dates)
    
    # Make semi-annual coupon payments starting from first coupon date until maturity
    coupon_dates = pd.date_range(start=first_coupon_date, end=maturity_date, freq='6M')
    
    # Adjust coupon payments and bond value based on the inflation index
    for date in coupon_dates:
        # Inflation adjustment factor: inflation index at coupon date / inflation index at issuance
        inflation_adjustment = inflation_index.loc[date] / inflation_index.loc[first_coupon_date]
        
        # Adjusted bond value and coupon payment
        adjusted_bond_value = bond_value * inflation_adjustment
        coupon_payment = adjusted_bond_value * (coupon_rate / 2)
        
        bond_payments[date] = coupon_payment
    
    # Add both the final coupon and the inflation-adjusted bond value at maturity
    final_inflation_adjustment = inflation_index.get(maturity_date, 1.0) / inflation_index.get(first_coupon_date, 1.0)
    adjusted_bond_value = bond_value * final_inflation_adjustment
    bond_payments[maturity_date] += adjusted_bond_value
    
    return bond_payments


# Initialize an empty DataFrame for the payment schedule
tips_schedule = pd.DataFrame(index=monthly_dates)

# Apply the function to each bond in the cleaned dataset and store the result in the payment schedule
for _, bond_row in crsp_tips.iterrows():
    bond_id = bond_row['CRSPID']
    tips_schedule[bond_id] = compute_payments_for_inflation_bond(bond_row, fred_data['CPI'])



'''
    Compute MWD 
'''

def compute_mwd(crsp_data, schedule, func, name, weight):
    
    if weight == True:
    
        ''' Generate duration matrix '''
        
        # Extract year and month from the dates
        years = monthly_dates.year
        months = monthly_dates.month
        
        # Create arrays for broadcasting
        year_diff = years.values[:, None] - years.values  # Difference in years
        month_diff = months.values[:, None] - months.values  # Difference in months
        
        # Calculate total month difference
        months_diff_array = year_diff * 12 + month_diff
        
        # Clip negative values to zero
        months_diff_array = np.clip(months_diff_array, 0, None)
        
        # Create a DataFrame with the result
        months_difference_table = pd.DataFrame(months_diff_array, index=monthly_dates, columns=monthly_dates)
        
        # Optionally, divide by 12 to convert to years
        year_diff_table = months_difference_table / 12
        
        # func is for parameterisation in the Zhao (2024) paper
        year_difference_table = year_diff_table.applymap(func)
        
        
    elif weight  == False:
        
        # Initialize the DataFrame with the date range as both index and columns
        year_difference_table = pd.DataFrame(index=monthly_dates, columns=monthly_dates)
        
        # Fill the DataFrame with the difference between the column and index dates
        for col in year_difference_table.columns:
            year_difference_table[col] = np.where((year_difference_table.index - col).days > 0, 1, 0)
        
        
    
    ''' Calculate MWD '''
    
    crsp_data.set_index('CRSPID', inplace=True)

    # Ensure both crsp_data['TDATDT'] and year_difference_table columns are in comparable formats
    issue_track = crsp_data['TDATDT'].values[:, None] <= year_difference_table.columns.values
    
    # Convert the result back into a DataFrame
    issue_track = pd.DataFrame(issue_track, index=crsp_data.index, columns=year_difference_table.columns)
    
    
    mwd_id = pd.DataFrame(index=monthly_dates)
    cutoff_date = pd.Timestamp('2024-01-01')
    
    
    for col_date in year_difference_table.columns:
        if col_date >= cutoff_date:
            break
        
        print(col_date)
        
        for crsp_id in schedule.columns:
            # Compute the dot product between the column of months_difference_table and the Total Payments column
            dot_prod = np.dot(year_difference_table[col_date], schedule[crsp_id]) * issue_track.at[crsp_id,col_date] 
        
            # Store the result
            mwd_id.at[col_date, crsp_id] = dot_prod
    
    mwd = pd.DataFrame(mwd_id.sum(axis=1), columns=[name])
    mwd = mwd.reindex(fred_data.index)
    mwd[name + '_GDP'] = mwd[name].div(fred_data['GDP'])
    
    crsp_data.reset_index(inplace=True)
    return mwd


# function y = x
def nothing(tau):
    return tau


# MWD
mwd_nom = compute_mwd(crsp_nominal, nominal_schedule, nothing, 'mwd_nom', True)
mwd_tips = compute_mwd(crsp_tips, tips_schedule, nothing, 'mwd_tips', True)

mwd_agg = pd.DataFrame(index=mwd_nom.index)
mwd_agg['mwd_agg'] = mwd_nom['mwd_nom'] + mwd_tips['mwd_tips']
mwd_agg['mwd_agg_GDP'] = mwd_nom['mwd_nom_GDP'] + mwd_tips['mwd_tips_GDP']


# Debt
debt_nom = compute_mwd(crsp_nominal, nominal_schedule, nothing, 'debt_nom', False)
debt_tips = compute_mwd(crsp_tips, tips_schedule, nothing, 'debt_tips', False)

debt_agg = pd.DataFrame(index=debt_nom.index)
debt_agg['debt_agg'] = debt_nom['debt_nom'] + debt_tips['debt_tips']
debt_agg['debt_agg_GDP'] = debt_nom['debt_nom_GDP'] + debt_tips['debt_tips_GDP']


# def plot_and_save(mwd,filename):
#     mwd.to_csv(f'output/{filename}.csv', index=True)
    
#     # Plotting
#     plt.figure(figsize=(10, 6))  # Set figure size
#     plt.plot(mwd.index, mwd[filename +'_GDP'], label='MWD to GDP')
    
#     plt.xlabel('Time')    # Add labels and title
#     plt.grid(True)        # Show grid for better readability
#     plt.legend()          # Display the legend
#     plt.show()            # Show the plot


# plot_and_save(mwd_nom,'mwd_nom')
# plot_and_save(mwd_tips,'mwd_tips')

'''
    Cashflow Weighted-Average-Maturity (WAM)
'''
# Merge all datasets sequentially on the index using reduce
maturity = pd.DataFrame(index = mwd_nom.index)
maturity['mat_nom'] = mwd_nom['mwd_nom'] / debt_nom['debt_nom']
maturity['mat_tips'] = mwd_tips['mwd_tips'] / debt_tips['debt_tips']
maturity['mat_agg'] = mwd_agg['mwd_agg'] / debt_agg['debt_agg']
maturity = maturity.loc['1999-01-31':'2023-12-31']



'''
    Export 
'''
# Load all datasets and ensure the index is used for merging
datasets = [
    mwd_nom, mwd_tips, mwd_agg, 
    debt_nom, debt_tips, debt_agg,
    maturity
]
# Merge all datasets sequentially on the index using reduce
treasury_mwd = reduce(lambda left, right: pd.merge(left, right, left_index=True, right_index=True, how='outer'), datasets)
treasury_mwd = treasury_mwd.loc['1999-01-31':'2023-12-31']
# Save the final combined dataset to a new CSV file
treasury_mwd.to_csv('output/treasury_mwd.csv', index=True)

