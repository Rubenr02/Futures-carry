import pandas as pd
import os

def clean_price_column(df):
    """Convert 'Price' column to float, stripping commas and other characters."""
    df['Price'] = (
        df['Price']
        .astype(str)
        .str.replace(',', '', regex=False)
        .str.replace('%', '', regex=False)
        .astype(float)
    )
    return df

def create_benchmarks(input_folder='Data', output_folder='benchmarks'):
    os.makedirs(output_folder, exist_ok=True)

    # Create S&P 500 benchmark
    try:
        sp500 = pd.read_csv(f'{input_folder}/S&P 500 Historical Data.csv', parse_dates=['Date'], dayfirst=True)
        sp500 = clean_price_column(sp500)
        sp500['Returns'] = sp500['Price'].pct_change()
        sp500[['Date', 'Returns']].to_csv(f'{output_folder}/sp500_benchmark.csv', index=False)
        print("Created S&P 500 benchmark")
    except FileNotFoundError:
        print("S&P 500 data missing - cannot create equity benchmark")
    except Exception as e:
        print(f"Error processing S&P 500: {e}")

    # Create US 10Y benchmark
    try:
        us10y = pd.read_csv(f'{input_folder}/United States 10-Year Bond Yield Historical Data.csv', parse_dates=['Date'], dayfirst=True)
        us10y = clean_price_column(us10y)
        us10y['Returns'] = us10y['Price'].pct_change()
        us10y[['Date', 'Returns']].to_csv(f'{output_folder}/us10y_benchmark.csv', index=False)
        print("Created US 10Y benchmark")
    except FileNotFoundError:
        print("US 10Y data missing - cannot create bond benchmark")
    except Exception as e:
        print(f" Error processing US 10Y: {e}")

if __name__ == '__main__':
    create_benchmarks()
