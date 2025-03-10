# treasury_mwd
Computes the maturity-weighted-debt (MWD) and MWD-to-GDP for both nominal Treasury debt, and real (TIPS) debt, using the CRSP Treasury bond database. 

Relevant citations:
* Zhao, N. (2024). _Preferred habitats across the real and nominal term structure of interest rates._ Unpublished manuscript. Downloaded from nimarzhao.com


## Instructions

1. Download monthly Treasury data from CRSP (subscription required). Rename file as **crsp_raw.csv** and place in **/data**
2. Run **clean_CRSP_TB.py** to generate **crsp_clean.csv**
3. Register for Federal Reserve Economic Data (FRED) API key
4. Run **clean_FRED.py** to generate **fred_clean.csv**
5. Compute MWD, Debt, and WAM using **compute_MWD.py**, saves to **/output/treasury_mwd.csv**
