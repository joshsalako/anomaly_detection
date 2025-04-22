# %%
import pandas as pd
import os
from datetime import datetime

# %%
# Function to parse date
def parse_date(date_str):
    try:
        return pd.to_datetime(date_str.strip(), format='%m/%d/%Y')
    except:
        try:
            return pd.to_datetime(date_str.strip(), format='%d/%m/%Y')
        except:
            return pd.NaT

# Function to clean and load CSV file
def load_csv(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path, sep=',', header=0)
    
    # Clean up column names
    df.columns = [col.strip() for col in df.columns]
    
    # Convert date to datetime
    df['Date'] = df['Date'].apply(lambda x: parse_date(x.strip('"') if isinstance(x, str) else x))
    
    return df
# %%
# Load exchange rate data
usd_kes = pd.read_csv('datasets/USD_KES Historical Data.csv', sep=',')
usd_ngn = pd.read_csv('datasets/USD_NGN Historical Data.csv', sep=',')
usd_zar = pd.read_csv('datasets/USD_ZAR Historical Data.csv', sep=',')

# Clean up exchange rate data
for df in [usd_kes, usd_ngn, usd_zar]:
    df.columns = [col.strip('"') for col in df.columns]
    df['Date'] = df['Date'].apply(lambda x: parse_date(x.strip('"') if isinstance(x, str) else x))
    df['Price'] = df['Price'].apply(lambda x: float(x.strip('"').replace(',', '')) if isinstance(x, str) else float(x))
# %%
# List of stock files
stock_files = ['datasets/ANG.csv', 'datasets/DANGCEM.csv', 
               'datasets/GTCO.csv', 'datasets/MTN.csv', 'datasets/MTNN.csv', 
               'datasets/NB.csv', 'datasets/NPN.csv', 'datasets/SBK.csv', 
               'datasets/SCOM.csv', 'datasets/SOL.csv', 'datasets/ZENITHBANK.csv'
               ]

# Dictionary mapping stock to currency
stock_currency = {
    'SCOM': 'KES',
    'MTNN': 'NGN', 'DANGCEM': 'NGN', 'GTCO': 'NGN', 'NB': 'NGN', 'ZENITHBANK': 'NGN',
    'ANG': 'ZAR', 'MTN': 'ZAR', 'NPN': 'ZAR', 'SBK': 'ZAR', 'SOL': 'ZAR'
}

# Dictionary mapping currency to exchange rate dataframe
exchange_rates = {
    'KES': usd_kes,
    'NGN': usd_ngn,
    'ZAR': usd_zar
}
# %%
# List to store dataframes
dfs = []

# Process each stock file
for file in stock_files:
    print(f"Processing {file}...")
    stock_name = file.split('.')[0]
    
    # Load stock data
    df = load_csv(file)
    
    # Determine currency
    currency = stock_currency.get(stock_name)
    
    if currency:
        # Get exchange rate data
        rate_df = exchange_rates[currency]
        
        # Create a date index for faster lookup
        rate_dict = dict(zip(rate_df['Date'], rate_df['Price']))
        
        # Function to convert price to USD
        def convert_to_usd(row, column):
            date = row['Date']
            price = row[column]
            
            # Find closest date if exact date not available
            if date in rate_dict:
                rate = rate_dict[date]
            else:
                # Find closest date before the given date
                available_dates = [d for d in rate_dict.keys() if d <= date]
                if available_dates:
                    closest_date = max(available_dates)
                    rate = rate_dict[closest_date]
                else:
                    # If no earlier date, use the earliest available date
                    closest_date = min(rate_dict.keys())
                    rate = rate_dict[closest_date]
            
            # Convert price to USD (divide by exchange rate)
            if currency in ['KES', 'NGN', 'ZAR']:
                return price / rate
            else:
                return price
        
        # Convert price columns to USD
        for col in ['Open', 'High', 'Low', 'Close']:
            df[col] = df.apply(lambda row: convert_to_usd(row, col), axis=1)
    
    # Append to list of dataframes
    dfs.append(df)
# %%
# Concatenate all dataframes
combined_df = pd.concat(dfs, ignore_index=True)

# Sort by date
combined_df = combined_df.sort_values(by='Date')
# %%
# Save to CSV
combined_df.to_csv('datasets/africa_stock_5yrs.csv', index=False)

print("Processing complete. Output saved to africa_stock_5yrs.csv")

# %%
