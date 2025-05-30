import pandas as pd
import os
from datetime import datetime

def clean_price_column(series):
    """Clean price column by removing commas and converting to float."""
    return series.apply(lambda val: float(val.replace(',', '')) if isinstance(val, str) and ',' in val else float(val))

def process_futures_data(filepath, output_folder='output'):
    try:
        print(f"[{datetime.now()}] Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.strip()

        # Clean and transform
        df['Price'] = clean_price_column(df['Price'])
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df = df.sort_values('Date').reset_index(drop=True)

        # Generate Front and Next prices
        df['Front'] = df['Price']
        df['Next'] = df['Price'].shift(-1)
        df = df.dropna(subset=['Next'])

        # Final result
        result = df[['Date', 'Front', 'Next']]

        # Ensure output folder exists
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, 'crude_oil_carry_ready.csv')
        result.to_csv(output_path, index=False)

        print(f"[{datetime.now()}] Cleaned data saved to: {output_path}")
        print(result.head())

    except Exception as e:
        print(f"[{datetime.now()}] ERROR: {e}")

# Run the processing
process_futures_data('Crude Oil WTI Futures Historical Data.csv')
