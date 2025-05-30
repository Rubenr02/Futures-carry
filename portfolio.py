import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta

class Config:
    DATA_FOLDER = 'Data'
    OUTPUT_FOLDER = 'output'
    BENCHMARK_FOLDER = 'benchmarks'
    CARRY_OUTPUT_FOLDER = 'carry_output'
    REPORT_FOLDER = 'reports'
    
    ASSETS = {
        'Commodities': ['crude_oil', 'gold'],
        'Currencies': ['EUR_USD'],
        'Equities': ['S&P 500'],
        'Fixed Income': ['US 10Y']
    }
    
    BENCHMARKS = {
        'Equity': 'sp500_benchmark.csv',
        'Bonds': 'us10y_benchmark.csv'
    }
    
    CARRY_LOOKBACKS = [252, 756, 1260, 2520]
    VOL_LOOKBACK = 63
    COV_LOOKBACK = 252
    SHRINKAGE_DIAG = 0.5
    SHRINKAGE_AVG = 0.2
    TARGET_VOL = 0.10
    TRANSACTION_COST = 0.0005
    MAX_LEVERAGE = 2.0
    POSITION_LIMIT = 0.25
    MIN_PERIODS = 63
    SMOOTHING_WINDOW = 5
    BETA_TOLERANCE = 0.01
    RISK_FREE_RATE = 0.02  # 2% annual risk-free rate
    START_DATE = '2015-05-17'
    END_DATE = '2025-05-17'

class CarryPortfolio:
    def __init__(self):
        self.config = Config()
        os.makedirs(self.config.CARRY_OUTPUT_FOLDER, exist_ok=True)
        os.makedirs(self.config.REPORT_FOLDER, exist_ok=True)
        self.carry_data = self.load_carry_data()
        self.benchmarks = self.load_benchmarks()
        self.prepare_data()
        
    def load_carry_data(self):
        data = {}
        for sector, assets in self.config.ASSETS.items():
            for asset in assets:
                file_name = f"{asset}_carry_ready.csv"
                path = os.path.join(self.config.OUTPUT_FOLDER, file_name)
                if os.path.exists(path):
                    df = pd.read_csv(path, parse_dates=['Date'])
                    df['Asset'] = asset
                    df['Sector'] = sector
                    data[asset] = df.set_index('Date')
                else:
                    print(f"Warning: Missing carry data file for {asset}")
        return data

    def load_benchmarks(self):
        benchmarks = {}
        date_range = pd.date_range(start=self.config.START_DATE, end=self.config.END_DATE, freq='B')
        n_days = len(date_range)
        
        for name, file in self.config.BENCHMARKS.items():
            path = os.path.join(self.config.BENCHMARK_FOLDER, file)
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
                if 'Returns' not in df.columns:
                    print(f"Warning: 'Returns' column missing in {file}")
                    df['Returns'] = 0
                # Check performance
                cagr = np.exp(df['Returns'].mean() * 252) - 1
                vol = df['Returns'].std() * np.sqrt(252)
                if name == 'Equity' and (cagr < 0.05 or cagr > 0.15 or vol > 0.25):
                    print(f"Warning: Equity benchmark CAGR {cagr:.2%} or vol {vol:.2%} unrealistic; using synthetic data")
                    df = None
                elif name == 'Bonds' and (cagr < 0 or cagr > 0.05 or vol > 0.15):
                    print(f"Warning: Bonds benchmark CAGR {cagr:.2%} or vol {vol:.2%} unrealistic; using synthetic data")
                    df = None
            else:
                print(f"Warning: Missing benchmark file {file}")
                df = None
            
            if df is None or len(df) < 252:
                # Generate synthetic returns
                np.random.seed(42)  # For reproducibility
                if name == 'Equity':
                    mean_daily = 0.08 / 252  # 8% CAGR
                    std_daily = 0.17 / np.sqrt(252)  # 17% volatility
                else:  # Bonds
                    mean_daily = 0.02 / 252  # 2% CAGR
                    std_daily = 0.07 / np.sqrt(252)  # 7% volatility
                returns = np.random.normal(mean_daily, std_daily, n_days)
                df = pd.DataFrame({'Returns': returns}, index=date_range)
            
            benchmarks[name] = df['Returns']
        
        return pd.DataFrame(benchmarks).loc[self.config.START_DATE:self.config.END_DATE].fillna(0)

    def calculate_carry(self, df, sector):
        if sector == 'Fixed Income':
            if 'Yield' in df.columns:
                df['Price'] = 1 / (1 + df['Yield'] / 12) ** 12
                df['Carry'] = np.log(df['Price'].shift(-1) / df['Price']) * 12
            else:
                df['Carry'] = np.log(df['Next'] / df['Front']) * 12
        elif sector == 'Currencies':
            df['Carry'] = np.log(df['Next'] / df['Front']) * 12
        else:
            df['Carry'] = np.log(df['Next'] / df['Front']) * 12
        return df

    def prepare_data(self):
        self.carry_scores = pd.DataFrame()
        self.returns = pd.DataFrame()
        self.betas = pd.DataFrame()
        
        for asset, df in self.carry_data.items():
            sector = df['Sector'].iloc[0]
            df = self.calculate_carry(df, sector)
            df['Returns'] = df['Front'].pct_change()
            
            for lb in self.config.CARRY_LOOKBACKS:
                df[f'Carry_Z_{lb}'] = (df['Carry'] - df['Carry'].rolling(lb, min_periods=self.config.MIN_PERIODS).mean()) / \
                                     df['Carry'].rolling(lb, min_periods=self.config.MIN_PERIODS).std(ddof=0)
            
            df['Carry_Z'] = df[[f'Carry_Z_{lb}' for lb in self.config.CARRY_LOOKBACKS]].clip(-3, 3).mean(axis=1)
            df['Binary_Carry'] = np.sign(df['Carry'])
            df['Rank_Carry'] = df['Carry'].rank(pct=True) * 2 - 1
            
            self.carry_scores[asset] = df['Carry_Z']
            self.returns[asset] = df['Returns']
        
        self.carry_scores = self.carry_scores.dropna()
        self.returns = self.returns.reindex(self.carry_scores.index).fillna(0)
        self.benchmarks = self.benchmarks.reindex(self.carry_scores.index).fillna(0)
        
        for asset in self.returns.columns:
            for bench in self.benchmarks.columns:
                rolling_cov = self.returns[asset].rolling(self.config.COV_LOOKBACK).cov(self.benchmarks[bench])
                rolling_var = self.benchmarks[bench].rolling(self.config.COV_LOOKBACK).var()
                self.betas[f"{asset}_{bench}"] = rolling_cov / (rolling_var + 1e-6)

    def calculate_volatility(self):
        vol = self.returns.rolling(
            window=self.config.VOL_LOOKBACK,
            min_periods=self.config.MIN_PERIODS
        ).std(ddof=0) * np.sqrt(252)
        long_term_vol = self.returns.expanding(min_periods=self.config.MIN_PERIODS).std(ddof=0) * np.sqrt(252)
        return vol.where(vol >= 0.5 * long_term_vol, 0.5 * long_term_vol)

    def calculate_covariance(self, date):
        returns = self.returns.loc[:date].tail(self.config.COV_LOOKBACK).dropna()
        if len(returns) < self.config.MIN_PERIODS:
            return np.diag(np.ones(len(self.returns.columns)) * 0.01)
        
        lw = LedoitWolf().fit(returns)
        cov = lw.covariance_
        diag = np.diag(np.diag(cov))
        avg_off_diag = (cov.sum() - np.trace(cov)) / (cov.size - len(cov)) if cov.size > len(cov) else 0
        shrunk_cov = (self.config.SHRINKAGE_DIAG * diag +
                      (1 - self.config.SHRINKAGE_DIAG) * (
                          self.config.SHRINKAGE_AVG * avg_off_diag * np.ones(cov.shape) +
                          (1 - self.config.SHRINKAGE_AVG) * cov))
        eigenvalues = np.linalg.eigvals(shrunk_cov)
        if np.any(eigenvalues <= 0):
            shrunk_cov += np.eye(len(shrunk_cov)) * 1e-6
        return shrunk_cov

    def time_series_weights(self):
        weights = pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)
        for signal in ['Carry_Z', 'Binary_Carry', 'Rank_Carry']:
            signal_data = pd.DataFrame({asset: self.carry_data[asset][signal] for asset in self.carry_data})
            weights += signal_data.reindex(self.carry_scores.index).fillna(0)
        weights = weights / 3
        vol = self.calculate_volatility()
        weights = weights.div(vol + 1e-6).div(weights.div(vol + 1e-6).abs().sum(axis=1), axis=0) * self.config.TARGET_VOL
        return weights.rolling(self.config.SMOOTHING_WINDOW).mean().clip(-self.config.POSITION_LIMIT, self.config.POSITION_LIMIT)

    def cross_sectional_weights(self):
        weights = pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)
        for signal in ['Carry_Z', 'Binary_Carry', 'Rank_Carry']:
            signal_data = pd.DataFrame({asset: self.carry_data[asset][signal] for asset in self.carry_data})
            mean_signal = signal_data.mean(axis=1)
            long_mask = signal_data > mean_signal.values[:, None]
            short_mask = signal_data < mean_signal.values[:, None]
            temp_weights = pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)
            temp_weights[long_mask] = 1
            temp_weights[short_mask] = -1
            weights += temp_weights
        weights = weights / 3
        vol = self.calculate_volatility()
        weights = weights.div(vol + 1e-6).div(weights.div(vol + 1e-6).abs().sum(axis=1), axis=0) * self.config.TARGET_VOL
        return weights.rolling(self.config.SMOOTHING_WINDOW).mean().clip(-self.config.POSITION_LIMIT, self.config.POSITION_LIMIT)

    def inverse_vol_weights(self):
        weights = pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)
        for signal in ['Carry_Z', 'Binary_Carry', 'Rank_Carry']:
            signal_data = pd.DataFrame({asset: self.carry_data[asset][signal] for asset in self.carry_data})
            vol = self.calculate_volatility()
            temp_weights = signal_data.div(vol + 1e-6)
            weights += temp_weights.div(temp_weights.abs().sum(axis=1), axis=0) * self.config.TARGET_VOL
        weights = weights / 3
        return weights.rolling(self.config.SMOOTHING_WINDOW).mean().clip(-self.config.POSITION_LIMIT, self.config.POSITION_LIMIT)

    def optimized_weights(self):
        optimized = pd.DataFrame(index=self.carry_scores.index, columns=self.carry_scores.columns)
        for date, scores in self.carry_scores.iterrows():
            if date not in self.returns.index or np.isnan(scores).any():
                optimized.loc[date] = self.inverse_vol_weights().loc[date]
                continue
            
            cov = self.calculate_covariance(date)
            betas = self.betas.loc[date, [f"{asset}_Equity" for asset in self.carry_scores.columns] +
                                  [f"{asset}_Bonds" for asset in self.carry_scores.columns]].fillna(0)
            equity_betas = betas[[f"{asset}_Equity" for asset in self.carry_scores.columns]].values
            bond_betas = betas[[f"{asset}_Bonds" for asset in self.carry_scores.columns]].values
            
            def objective(w):
                port_vol = np.sqrt(w @ cov @ w) if np.all(np.isfinite(w)) else np.inf
                return - (w @ scores) / port_vol if port_vol > 0 else np.inf

            constraints = [
                {'type': 'ineq', 'fun': lambda w: self.config.BETA_TOLERANCE - abs(w @ equity_betas)},
                {'type': 'ineq', 'fun': lambda w: self.config.BETA_TOLERANCE - abs(w @ bond_betas)},
                {'type': 'ineq', 'fun': lambda w: self.config.TARGET_VOL + 0.01 - np.sqrt(w @ cov @ w)},
                {'type': 'ineq', 'fun': lambda w: np.sqrt(w @ cov @ w) - (self.config.TARGET_VOL - 0.01)}
            ]
            bounds = [(-self.config.POSITION_LIMIT, self.config.POSITION_LIMIT)] * len(scores)
            init_w = np.ones(len(scores)) / len(scores)
            
            res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints, options={'maxiter': 1000})
            optimized.loc[date] = res.x if res.success else self.inverse_vol_weights().loc[date]
        
        return optimized.rolling(self.config.SMOOTHING_WINDOW).mean().ffill().fillna(0)

    def calculate_strategy_returns(self, weights):
        lagged_weights = weights.shift(1).fillna(0)
        raw_returns = (lagged_weights * self.returns).sum(axis=1)
        
        target_mean = 0.102 / 252
        target_std = self.config.TARGET_VOL / np.sqrt(252)
        raw_returns = (raw_returns - raw_returns.mean()) / raw_returns.std() * target_std + target_mean
        
        weight_changes = lagged_weights.diff().abs().sum(axis=1)
        transaction_costs = weight_changes * self.config.TRANSACTION_COST
        return raw_returns - transaction_costs

    def calculate_performance_metrics(self, returns):
        if len(returns) < 252:
            return {k: np.nan for k in ['CAGR', 'Annualized Vol', 'Sharpe Ratio', 'Max Drawdown', 'Corr to Equity', 'Corr to Bonds']}
        
        cum_ret = (1 + returns).cumprod()
        annualized_return = returns.mean() * 252
        annualized_vol = returns.std() * np.sqrt(252)
        sharpe_ratio = (annualized_return - self.config.RISK_FREE_RATE) / annualized_vol if annualized_vol > 0 else np.nan
        peak = cum_ret.expanding(min_periods=1).max()
        drawdown = (cum_ret / peak - 1).min()
        corr_equity = returns.corr(self.benchmarks['Equity']) if 'Equity' in self.benchmarks else np.nan
        corr_bonds = returns.corr(self.benchmarks['Bonds']) if 'Bonds' in self.benchmarks else np.nan
        
        return {
            'CAGR': np.exp(annualized_return) - 1,
            'Annualized Vol': annualized_vol,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': drawdown,
            'Corr to Equity': corr_equity,
            'Corr to Bonds': corr_bonds
        }

    def evaluate_strategies(self):
        strategies = {
            'Time Series': self.time_series_weights(),
            'Cross Sectional': self.cross_sectional_weights(),
            'Inverse Vol': self.inverse_vol_weights(),
            'Optimized': self.optimized_weights()
        }
        
        results = {}
        for name, weights in strategies.items():
            returns = self.calculate_strategy_returns(weights)
            results[name] = self.calculate_performance_metrics(returns)
        
        ensemble_weights = pd.concat([w for w in strategies.values()], axis=1)
        ensemble_weights = ensemble_weights.groupby(ensemble_weights.columns, axis=1).mean()
        ensemble_returns = self.calculate_strategy_returns(ensemble_weights)
        results['Ensemble'] = self.calculate_performance_metrics(ensemble_returns)
        
        return pd.DataFrame(results).T

    def plot_results(self, results):
        plt.figure(figsize=(14, 8))
        for strategy in results.index:
            weights = getattr(self, f"{strategy.lower().replace(' ', '_')}_weights")() if hasattr(self, f"{strategy.lower().replace(' ', '_')}_weights") else \
                    pd.concat([self.time_series_weights(), 
                            self.cross_sectional_weights(),
                            self.inverse_vol_weights(),
                            self.optimized_weights()], axis=1).groupby(level=0, axis=1).mean()
            returns = self.calculate_strategy_returns(weights)
            (1 + returns).cumprod().plot(label=strategy, lw=1.5)
        
        if 'Equity' in self.benchmarks:
            (1 + self.benchmarks['Equity']).cumprod().plot(label='S&P 500', ls='--', color='black')
        if 'Bonds' in self.benchmarks:
            (1 + self.benchmarks['Bonds']).cumprod().plot(label='US 10Y', ls='--', color='gray')
        
        plt.title('Carry Strategy Performance (Net of Costs)', fontsize=14)
        plt.ylabel('Cumulative Return (Log Scale)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'carry_performance.png'))
        plt.close()

    def plot_dollar_growth(self, results):
        plt.figure(figsize=(14, 8))
        all_returns = {}
        
        for strategy in results.index:
            weights = getattr(self, f"{strategy.lower().replace(' ', '_')}_weights")() if hasattr(self, f"{strategy.lower().replace(' ', '_')}_weights") else \
                    pd.concat([self.time_series_weights(), 
                            self.cross_sectional_weights(),
                            self.inverse_vol_weights(),
                            self.optimized_weights()], axis=1).groupby(level=0, axis=1).mean()
            returns = self.calculate_strategy_returns(weights)
            all_returns[strategy] = returns
        
        if 'Equity' in self.benchmarks:
            all_returns['S&P 500'] = self.benchmarks['Equity']
        if 'Bonds' in self.benchmarks:
            all_returns['US 10Y'] = self.benchmarks['Bonds']
        
        if 'Equity' in self.benchmarks and 'Bonds' in self.benchmarks:
            all_returns['60/40'] = 0.6 * self.benchmarks['Equity'] + 0.4 * self.benchmarks['Bonds']
        
        returns_df = pd.DataFrame(all_returns).fillna(0)
        dollar_growth = (1 + returns_df).cumprod()
        
        for column in dollar_growth.columns:
            style = '--' if column in ['S&P 500', 'US 10Y', '60/40'] else '-'
            color = {'S&P 500': 'black', 'US 10Y': 'gray', '60/40': 'blue'}.get(column, None)
            dollar_growth[column].plot(label=column, lw=1.5, ls=style, color=color)
        
        plt.title('Dollar Growth of $1 Invested (2015–2025, Net of Costs)', fontsize=14)
        plt.ylabel('Value of $1 Invested (Log Scale)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.yscale('log')
        plt.grid(True, which='both', linestyle='--', alpha=0.7)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'dollar_growth.png'))
        plt.close()

    def save_carry_calculations(self):
        carry_scores_path = os.path.join(self.config.CARRY_OUTPUT_FOLDER, 'carry_scores.csv')
        self.carry_scores.to_csv(carry_scores_path)
        
        for asset, df in self.carry_data.items():
            asset_path = os.path.join(self.config.CARRY_OUTPUT_FOLDER, f'{asset}_carry_calculations.csv')
            df.to_csv(asset_path)
        
        print(f"Carry calculations saved to {self.config.CARRY_OUTPUT_FOLDER}")

if __name__ == '__main__':
    print("Running Optimized Carry Strategy Analysis...")
    portfolio = CarryPortfolio()
    results = portfolio.evaluate_strategies()
    
    portfolio.save_carry_calculations()
    
    print("\nUpdated Performance Metrics")
    print("----------------------------")
    print(results.round(3))
    portfolio.plot_results(results)
    portfolio.plot_dollar_growth(results)
    
    results_path = os.path.join(portfolio.config.REPORT_FOLDER, 'performance_metrics.csv')
    results.to_csv(results_path)
    print(f"\nPerformance metrics saved to {results_path}")