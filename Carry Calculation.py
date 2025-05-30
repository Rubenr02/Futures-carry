import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from matplotlib.ticker import PercentFormatter

# ---------- Configuration ----------
INPUT_FOLDER = 'carry_output'
BENCHMARK_FOLDER = 'benchmarks'
ASSET_FILES = {
    'Crude Oil': 'crude_oil_carry_ready.csv',
    'Gold': 'gold_carry_ready.csv',
    'EUR/USD': 'EUR_USD_carry_ready.csv',
    'S&P 500': 'S&P 500_carry_ready.csv',
    'US 10Y': 'US 10Y_carry_ready.csv'
}
SECTOR_MAP = {
    'Commodities': ['Crude Oil', 'Gold'],
    'Equities': ['S&P 500'],
    'Fixed Income': ['US 10Y'],
    'Currencies': ['EUR/USD']
}

# ---------- Core Functions ----------
def calculate_carry(df):
    df = df.copy()
    df['Carry'] = np.log(df['Front'] / df['Next']) * 12
    df['Returns'] = df['Front'].pct_change()
    return df.dropna()

def volatility_weighting(returns, lookback=63):
    vol = returns.rolling(lookback).std() * np.sqrt(252)
    weights = 1 / vol
    return weights.div(weights.sum(axis=1), axis=0)

def portfolio_optimization(carry_scores, returns, lookback=252, shrinkage=0.5):
    cov_estimator = LedoitWolf()
    optimized_weights = pd.DataFrame(index=carry_scores.index, columns=carry_scores.columns, dtype=float)
    
    for date in carry_scores.index[lookback:]:
        scores = carry_scores.loc[date].values
        ret_window = returns.loc[:date]
        
        if len(ret_window) < lookback or ret_window.isnull().values.any():
            continue
        
        ret_window = ret_window.iloc[-lookback:]
        cov = cov_estimator.fit(ret_window).covariance_
        shrunk_cov = shrinkage * np.diag(np.diag(cov)) + (1 - shrinkage) * cov
        
        def objective(w):
            return -w @ scores + 0.5 * w @ shrunk_cov @ w
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(-1, 1)] * len(scores)
        init_w = np.ones(len(scores)) / len(scores)
        
        result = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            optimized_weights.loc[date] = result.x
        else:
            optimized_weights.loc[date] = np.nan
    
    return optimized_weights.ffill().dropna()

def beta_neutral_strategy(carry_scores, returns, benchmarks, lookback=252):
    all_data = pd.concat([returns, benchmarks], axis=1).dropna()
    beta_neutral_weights = pd.DataFrame(index=carry_scores.index, columns=carry_scores.columns, dtype=float)
    
    for date in carry_scores.index[lookback:]:
        scores = carry_scores.loc[date].values
        window_data = all_data.loc[:date]
        
        if len(window_data) < lookback or window_data.isnull().values.any():
            continue
        
        window_data = window_data.iloc[-lookback:]
        cov_matrix = window_data.cov()
        cov_returns = cov_matrix.loc[carry_scores.columns, carry_scores.columns]
        
        # Get covariance with benchmarks
        cov_with_equity = cov_matrix.loc[carry_scores.columns, 'Equity'].values
        cov_with_bonds = cov_matrix.loc[carry_scores.columns, 'Bonds'].values
        
        def objective(w):
            return -w @ scores
        
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'eq', 'fun': lambda w: w @ cov_with_equity},
            {'type': 'eq', 'fun': lambda w: w @ cov_with_bonds}
        ]
        
        bounds = [(-1, 1)] * len(scores)
        init_w = np.ones(len(scores)) / len(scores)
        
        result = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            beta_neutral_weights.loc[date] = result.x
        else:
            beta_neutral_weights.loc[date] = np.nan
    
    return beta_neutral_weights.ffill().dropna()

# ---------- Data Loading ----------
def load_data():
    carry_data = {}
    for asset, file in ASSET_FILES.items():
        path = os.path.join(INPUT_FOLDER, file)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['Date'], dayfirst=True)
            carry_data[asset] = calculate_carry(df).set_index('Date')
    
    benchmarks = {}
    for bench, file in [('Equity', 'sp500_benchmark.csv'), ('Bonds', 'us10y_benchmark.csv')]:
        path = os.path.join(BENCHMARK_FOLDER, file)
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=['Date'], dayfirst=True)
            benchmarks[bench] = df.set_index('Date')['Returns']
    
    benchmarks = pd.DataFrame(benchmarks)
    return carry_data, benchmarks

# ---------- Strategies ----------
def time_series_strategy(carry_data):
    positions = pd.DataFrame({asset: np.sign(df['Carry']) for asset, df in carry_data.items()})
    return positions.dropna()

def cross_sectional_strategy(carry_data):
    carry_df = pd.DataFrame({asset: df['Carry'] for asset, df in carry_data.items()})
    ranks = carry_df.rank(axis=1, ascending=False)
    threshold = len(carry_df.columns) // 2
    return (ranks <= threshold).astype(int) - (ranks > threshold).astype(int)

# ---------- Performance Analysis ----------
def analyze_performance(strategy_returns, benchmarks):
    print("\nPerformance Summary:")
    results = {}
    for name, returns in strategy_returns.items():
        returns = returns.dropna()
        cum_ret = (1 + returns).cumprod()
        sharpe = np.sqrt(252) * returns.mean() / returns.std() if returns.std() > 0 else np.nan
        max_dd = (cum_ret / cum_ret.cummax() - 1).min()
        cagr = (cum_ret.iloc[-1]) ** (252 / len(cum_ret)) - 1 if len(cum_ret) > 0 else np.nan
        corr_equity = returns.corr(benchmarks['Equity']) if 'Equity' in benchmarks else np.nan
        corr_bonds = returns.corr(benchmarks['Bonds']) if 'Bonds' in benchmarks else np.nan
        
        results[name] = {
            'Sharpe': sharpe,
            'Max Drawdown': max_dd,
            'CAGR': cagr,
            'Corr Equity': corr_equity,
            'Corr Bonds': corr_bonds
        }
    
    df_results = pd.DataFrame(results).T.round(3)
    print(df_results)
    
    # Rolling Sharpe plot
    rolling_sharpe = strategy_returns.rolling(63).apply(lambda x: np.sqrt(252)*x.mean()/x.std() if x.std() > 0 else np.nan)
    plt.figure(figsize=(12, 6))
    rolling_sharpe.plot(title='Rolling 3-Month Sharpe Ratio')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.show()
    
    # Cumulative returns plot
    plt.figure(figsize=(12, 6))
    for name, returns in strategy_returns.items():
        (1 + returns.fillna(0)).cumprod().plot(label=name)
    if 'Equity' in benchmarks:
        (1 + benchmarks['Equity'].fillna(0)).cumprod().plot(label='S&P 500', ls='--', color='black')
    plt.title('Strategy and Benchmark Cumulative Returns')
    plt.legend()
    plt.show()
    
    # Correlation heatmap
    corr = strategy_returns.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, fmt=".2f")
    plt.title('Strategy Return Correlation')
    plt.show()

# ---------- Sector Breakdown ----------
def sector_returns(carry_data):
    sector_rets = {}
    for sector, assets in SECTOR_MAP.items():
        valid_assets = [a for a in assets if a in carry_data]
        if valid_assets:
            rets = pd.DataFrame({a: carry_data[a]['Returns'] for a in valid_assets})
            sector_rets[sector] = rets.mean(axis=1)
    return pd.DataFrame(sector_rets)

def worst_quarter_analysis(strategy_returns):
    quarterly_returns = strategy_returns.resample('Q').apply(lambda x: (1 + x).prod() - 1)
    worst_quarters = quarterly_returns.min()
    print("\nWorst Quarterly Returns:")
    print(worst_quarters.round(4))

# ---------- Main Execution ----------
def main():
    carry_data, benchmarks = load_data()
    
    # Combine carry scores into DataFrame aligned on index
    carry_df = pd.DataFrame({asset: df['Carry'] for asset, df in carry_data.items()})
    returns_df = pd.DataFrame({asset: df['Returns'] for asset, df in carry_data.items()})
    
    # Strategies
    ts_pos = time_series_strategy(carry_data)
    cs_pos = cross_sectional_strategy(carry_data)
    vol_weights = volatility_weighting(returns_df)
    opt_weights = portfolio_optimization(carry_df, returns_df)
    beta_neutral_weights = beta_neutral_strategy(carry_df, returns_df, benchmarks)
    
    # Compute strategy returns
    strategy_returns = {}
    
    # Time series strategy returns
    ts_ret = (returns_df * ts_pos).mean(axis=1)
    strategy_returns['Time Series'] = ts_ret
    
    # Cross sectional strategy returns
    cs_ret = (returns_df * cs_pos).mean(axis=1)
    strategy_returns['Cross Sectional'] = cs_ret
    
    # Volatility-weighted returns
    vw_ret = (returns_df * vol_weights).mean(axis=1)
    strategy_returns['Volatility Weighted'] = vw_ret
    
    # Optimized portfolio returns
    opt_ret = (returns_df * opt_weights).mean(axis=1)
    strategy_returns['Optimized Portfolio'] = opt_ret
    
    # Beta neutral portfolio returns
    bn_ret = (returns_df * beta_neutral_weights).mean(axis=1)
    strategy_returns['Beta Neutral'] = bn_ret
    
    # Ensemble strategy (average of all above strategies)
    ensemble = pd.DataFrame(strategy_returns).mean(axis=1)
    strategy_returns['Ensemble'] = ensemble
    
    # Add benchmarks returns
    if 'Equity' in benchmarks:
        strategy_returns['S&P 500'] = benchmarks['Equity']
    if 'Bonds' in benchmarks:
        strategy_returns['US 10Y'] = benchmarks['Bonds']
    
    strategy_returns = pd.DataFrame(strategy_returns)
    
    # Performance analysis
    analyze_performance(strategy_returns, benchmarks)
    
    # Sector breakdown
    sector_rets = sector_returns(carry_data)
    plt.figure(figsize=(12,6))
    (1 + sector_rets).cumprod().plot()
    plt.title('Sector-level Cumulative Returns')
    plt.ylabel('Cumulative Return')
    plt.grid(True)
    plt.show()
    
    # Worst quarter analysis
    worst_quarter_analysis(strategy_returns)

if __name__ == '__main__':
    main()
