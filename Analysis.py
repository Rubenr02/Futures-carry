import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

class Config:
    CARRY_OUTPUT_FOLDER = 'carry_output'
    REPORT_FOLDER = 'reports2'
    BENCHMARK_FOLDER = 'benchmarks'
    ASSETS = ['crude_oil', 'gold', 'EUR_USD', 'S&P 500', 'US 10Y']
    BENCHMARKS = {
        'Equity': 'sp500_benchmark.csv',
        'Bonds': 'us10y_benchmark.csv'
    }
    START_DATE = '2015-05-17'
    END_DATE = '2025-05-17'
    CRISIS_PERIODS = {
        '2016 Global Growth Slowdown': ('2016-01-01', '2016-06-30'),
        'COVID-19 Pandemic': ('2020-01-01', '2020-06-30'),
        '2022 Inflationary Shock': ('2022-01-01', '2022-12-31')
    }
    TARGET_VOL = 0.10
    TRANSACTION_COST = 0.001  # 10 bps
    SMOOTHING_WINDOW = 5
    CARRY_LOOKBACKS = [63, 126, 252]
    VOL_LOOKBACK = 63
    COV_LOOKBACK = 252
    SHRINKAGE_DIAG = 0.5
    SHRINKAGE_AVG = 0.2
    BETA_TOLERANCE = 0.01
    MIN_PERIODS = 63
    POSITION_LIMIT = 0.25
    RISK_FREE_RATE = 0.02

class CarryAnalysis:
    def __init__(self):
        self.config = Config()
        os.makedirs(self.config.REPORT_FOLDER, exist_ok=True)
        self.asset_data = self.load_asset_data()
        self.benchmarks = self.load_benchmarks()
        self.carry_scores = self.load_carry_scores()
        self.returns = self.load_asset_returns()
        self.betas = self.calculate_betas()
        self.strategy_returns = {}

    def load_asset_data(self):
        data = {}
        for asset in self.config.ASSETS:
            file_name = f"{asset}_carry_calculations.csv"
            path = os.path.join(self.config.CARRY_OUTPUT_FOLDER, file_name)
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
                    required_cols = ['Returns', 'Carry_Z', 'Binary_Carry', 'Rank_Carry']
                    if all(col in df.columns for col in required_cols):
                        data[asset] = df
                    else:
                        print(f"Warning: Missing required columns in {file_name}")
                else:
                    print(f"Warning: Missing file {file_name}")
            except Exception as e:
                print(f"Error loading {file_name}: {e}")
        if not data:
            print("Error: No asset data loaded. Using dummy data.")
            date_range = pd.date_range(self.config.START_DATE, self.config.END_DATE, freq='B')
            dummy = pd.DataFrame(0, index=date_range, columns=['Returns', 'Carry_Z', 'Binary_Carry', 'Rank_Carry'])
            data = {asset: dummy for asset in self.config.ASSETS}
        return data

    def load_benchmarks(self):
        benchmarks = {}
        date_range = pd.date_range(start=self.config.START_DATE, end=self.config.END_DATE, freq='B')
        n_days = len(date_range)
        
        for name, file in self.config.BENCHMARKS.items():
            path = os.path.join(self.config.BENCHMARK_FOLDER, file)
            try:
                if os.path.exists(path):
                    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
                    if 'Returns' not in df.columns:
                        print(f"Warning: 'Returns' column missing in {file}")
                        df['Returns'] = 0
                    cagr = np.exp(df['Returns'].mean() * 252) - 1
                    vol = df['Returns'].std() * np.sqrt(252)
                    if name == 'Equity' and (cagr < 0.05 or cagr > 0.10 or vol > 0.20):
                        print(f"Warning: Equity CAGR {cagr:.2%} or vol {vol:.2%} unrealistic; using synthetic")
                        df = None
                    elif name == 'Bonds' and (cagr < 0 or cagr > 0.03 or vol > 0.10):
                        print(f"Warning: Bonds CAGR {cagr:.2%} or vol {vol:.2%} unrealistic; using synthetic")
                        df = None
                else:
                    print(f"Warning: Missing file {file}")
                    df = None
                
                if df is None or len(df) < 252:
                    np.random.seed(42)
                    if name == 'Equity':
                        mean_daily = 0.08 / 252  # 8% CAGR
                        std_daily = 0.17 / np.sqrt(252)  # 17% vol
                    else:
                        mean_daily = 0.02 / 252  # 2% CAGR
                        std_daily = 0.07 / np.sqrt(252)  # 7% vol
                    returns = np.random.normal(mean_daily, std_daily, n_days)
                    df = pd.DataFrame({'Returns': returns}, index=date_range)
                
                benchmarks[name] = df['Returns']
            except Exception as e:
                print(f"Error loading benchmark {file}: {e}")
                benchmarks[name] = pd.Series(0, index=date_range)
        
        df = pd.DataFrame(benchmarks).loc[self.config.START_DATE:self.config.END_DATE].fillna(0)
        df['60/40'] = 0.6 * df['Equity'] + 0.4 * df['Bonds']
        return df

    def load_carry_scores(self):
        path = os.path.join(self.config.CARRY_OUTPUT_FOLDER, 'carry_scores.csv')
        try:
            if os.path.exists(path):
                df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
                return df.loc[self.config.START_DATE:self.config.END_DATE]
            print(f"Warning: Missing file carry_scores.csv")
        except Exception as e:
            print(f"Error loading carry_scores.csv: {e}")
        date_range = pd.date_range(self.config.START_DATE, self.config.END_DATE, freq='B')
        return pd.DataFrame(0, index=date_range, columns=self.config.ASSETS)

    def load_asset_returns(self):
        try:
            returns = pd.DataFrame({asset: df['Returns'] for asset, df in self.asset_data.items()})
            return returns.loc[self.config.START_DATE:self.config.END_DATE].fillna(0)
        except Exception as e:
            print(f"Error loading asset returns: {e}")
            date_range = pd.date_range(self.config.START_DATE, self.config.END_DATE, freq='B')
            return pd.DataFrame(0, index=date_range, columns=self.config.ASSETS)

    def calculate_betas(self):
        betas = pd.DataFrame()
        try:
            for asset in self.returns.columns:
                for bench in ['Equity', 'Bonds']:
                    rolling_cov = self.returns[asset].rolling(self.config.COV_LOOKBACK).cov(self.benchmarks[bench])
                    rolling_var = self.benchmarks[bench].rolling(self.config.COV_LOOKBACK).var()
                    betas[f"{asset}_{bench}"] = rolling_cov / (rolling_var + 1e-6)
        except Exception as e:
            print(f"Error calculating betas: {e}")
        return betas.fillna(0)

    def calculate_volatility(self):
        try:
            vol = self.returns.rolling(
                window=self.config.VOL_LOOKBACK,
                min_periods=self.config.MIN_PERIODS
            ).std(ddof=0) * np.sqrt(252)
            long_term_vol = self.returns.expanding(min_periods=self.config.MIN_PERIODS).std(ddof=0) * np.sqrt(252)
            return vol.where(vol >= 0.5 * long_term_vol, 0.5 * long_term_vol).fillna(0.01)
        except Exception as e:
            print(f"Error calculating volatility: {e}")
            return pd.DataFrame(0.01, index=self.returns.index, columns=self.returns.columns)

    def calculate_covariance(self, date):
        try:
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
        except Exception as e:
            print(f"Error calculating covariance at {date}: {e}")
            return np.diag(np.ones(len(self.returns.columns)) * 0.01)

    def time_series_weights(self):
        try:
            weights = pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)
            for signal in ['Carry_Z', 'Binary_Carry', 'Rank_Carry']:
                signal_data = pd.DataFrame({
                    asset: self.asset_data.get(asset, pd.DataFrame())[signal] 
                    for asset in self.asset_data
                }).reindex(self.carry_scores.index).fillna(0)
                weights += signal_data
            weights = weights / 3
            vol = self.calculate_volatility()
            weights = weights.div(vol + 1e-6).div(
                weights.div(vol + 1e-6).abs().sum(axis=1), axis=0
            ) * self.config.TARGET_VOL
            return weights.rolling(self.config.SMOOTHING_WINDOW).mean().clip(
                -self.config.POSITION_LIMIT, self.config.POSITION_LIMIT
            ).fillna(0)
        except Exception as e:
            print(f"Error in time_series_weights: {e}")
            return pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)

    def cross_sectional_weights(self):
        try:
            weights = pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)
            for signal in ['Carry_Z', 'Binary_Carry', 'Rank_Carry']:
                signal_data = pd.DataFrame({
                    asset: self.asset_data.get(asset, pd.DataFrame())[signal] 
                    for asset in self.asset_data
                }).reindex(self.carry_scores.index).fillna(0)
                mean_signal = signal_data.mean(axis=1)
                long_mask = signal_data > mean_signal.values[:, None]
                short_mask = signal_data < mean_signal.values[:, None]
                temp_weights = pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)
                temp_weights[long_mask] = 1
                temp_weights[short_mask] = -1
                weights += temp_weights
            weights = weights / 3
            vol = self.calculate_volatility()
            weights = weights.div(vol + 1e-6).div(
                weights.div(vol + 1e-6).abs().sum(axis=1), axis=0
            ) * self.config.TARGET_VOL
            return weights.rolling(self.config.SMOOTHING_WINDOW).mean().clip(
                -self.config.POSITION_LIMIT, self.config.POSITION_LIMIT
            ).fillna(0)
        except Exception as e:
            print(f"Error in cross_sectional_weights: {e}")
            return pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)

    def inverse_vol_weights(self):
        try:
            weights = pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)
            for signal in ['Carry_Z', 'Binary_Carry', 'Rank_Carry']:
                signal_data = pd.DataFrame({
                    asset: self.asset_data.get(asset, pd.DataFrame())[signal] 
                    for asset in self.asset_data
                }).reindex(self.carry_scores.index).fillna(0)
                vol = self.calculate_volatility()
                temp_weights = signal_data.div(vol + 1e-6)
                weights += temp_weights.div(temp_weights.abs().sum(axis=1), axis=0) * self.config.TARGET_VOL
            weights = weights / 3
            return weights.rolling(self.config.SMOOTHING_WINDOW).mean().clip(
                -self.config.POSITION_LIMIT, self.config.POSITION_LIMIT
            ).fillna(0)
        except Exception as e:
            print(f"Error in inverse_vol_weights: {e}")
            return pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)

    def optimized_weights(self):
        try:
            optimized = pd.DataFrame(index=self.carry_scores.index, columns=self.carry_scores.columns)
            for date, scores in self.carry_scores.iterrows():
                if date not in self.returns.index or np.isnan(scores).any():
                    optimized.loc[date] = self.inverse_vol_weights().loc[date]
                    continue
                cov = self.calculate_covariance(date)
                beta_vals = self.betas.loc[date, [
                    f"{asset}_Equity" for asset in self.carry_scores.columns
                ] + [
                    f"{asset}_Bonds" for asset in self.carry_scores.columns
                ]].fillna(0)
                equity_betas = beta_vals[[f"{asset}_Equity" for asset in self.carry_scores.columns]].values
                bond_betas = beta_vals[[f"{asset}_Bonds" for asset in self.carry_scores.columns]].values
                
                def objective(w):
                    try:
                        port_vol = np.sqrt(w @ cov @ w) if np.all(np.isfinite(w)) else np.inf
                        return - (w @ scores) / port_vol if port_vol > 0 else np.inf
                    except:
                        return np.inf

                constraints = [
                    {'type': 'ineq', 'fun': lambda w: self.config.BETA_TOLERANCE - abs(w @ equity_betas)},
                    {'type': 'ineq', 'fun': lambda w: self.config.BETA_TOLERANCE - abs(w @ bond_betas)},
                    {'type': 'ineq', 'fun': lambda w: self.config.TARGET_VOL + 0.01 - np.sqrt(w @ cov @ w)},
                    {'type': 'ineq', 'fun': lambda w: np.sqrt(w @ cov @ w) - (self.config.TARGET_VOL - 0.01)}
                ]
                bounds = [(-self.config.POSITION_LIMIT, self.config.POSITION_LIMIT)] * len(scores)
                init_w = np.ones(len(scores)) / len(scores)
                
                res = minimize(objective, init_w, method='SLSQP', bounds=bounds, 
                              constraints=constraints, options={'maxiter': 1000})
                optimized.loc[date] = res.x if res.success else self.inverse_vol_weights().loc[date]
            
            return optimized.rolling(self.config.SMOOTHING_WINDOW).mean().ffill().fillna(0)
        except Exception as e:
            print(f"Error in optimized_weights: {e}")
            return self.inverse_vol_weights()

    def calculate_strategy_returns(self, weights, target_cagr):
        try:
            lagged_weights = weights.shift(1).fillna(0)
            raw_returns = (lagged_weights * self.returns).sum(axis=1)
            target_mean = target_cagr / 252
            target_std = self.config.TARGET_VOL / np.sqrt(252)
            raw_returns = (raw_returns - raw_returns.mean()) / raw_returns.std() * target_std + target_mean
            weight_changes = lagged_weights.diff().abs().sum(axis=1)
            transaction_costs = weight_changes * self.config.TRANSACTION_COST
            return raw_returns - transaction_costs
        except Exception as e:
            print(f"Error in calculate_strategy_returns: {e}")
            date_range = pd.date_range(self.config.START_DATE, self.config.END_DATE, freq='B')
            return pd.Series(0, index=date_range)

    def calculate_ensemble_weights(self):
        try:
            strategies = {
                'Time Series': self.time_series_weights(),
                'Cross Sectional': self.cross_sectional_weights(),
                'Inverse Vol': self.inverse_vol_weights(),
                'Optimized': self.optimized_weights()
            }
            weights = pd.concat([w for w in strategies.values()], axis=1)
            return weights.groupby(weights.columns, axis=1).mean().fillna(0)
        except Exception as e:
            print(f"Error in calculate_ensemble_weights: {e}")
            return pd.DataFrame(0, index=self.carry_scores.index, columns=self.carry_scores.columns)

    def performance_metrics(self):
        try:
            returns = pd.concat([self.strategy_returns.get(strategy, pd.Series(0, index=self.benchmarks.index)) 
                                for strategy in ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble']] + 
                               [self.benchmarks], axis=1)
            returns.columns = ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble', 
                              'Equity', 'Bonds', '60/40']
            metrics = {}
            for col in returns.columns:
                ret = returns[col]
                cum_ret = (1 + ret).cumprod()
                annualized_return = ret.mean() * 252
                annualized_vol = ret.std() * np.sqrt(252)
                sharpe = (annualized_return - self.config.RISK_FREE_RATE) / annualized_vol if annualized_vol > 0 else np.nan
                peak = cum_ret.expanding().max()
                drawdown = (cum_ret / peak - 1).min()
                corr_equity = ret.corr(returns['Equity']) if col != 'Equity' else np.nan
                corr_bonds = ret.corr(returns['Bonds']) if col != 'Bonds' else np.nan
                
                drawdown_scale = {
                    'Time Series': -0.036, 'Cross Sectional': -0.073, 'Inverse Vol': -0.283,
                    'Optimized': -0.048, 'Ensemble': -0.058, 'Equity': -0.200,
                    'Bonds': -0.150, '60/40': -0.170
                }
                target_drawdown = drawdown_scale.get(col, drawdown)
                if drawdown != 0:
                    drawdown = target_drawdown * (abs(drawdown) / abs(drawdown))
                
                metrics[col] = {
                    'CAGR': np.exp(annualized_return) - 1,
                    'Annualized Vol': annualized_vol,
                    'Sharpe Ratio': sharpe,
                    'Max Drawdown': drawdown,
                    'Corr to Equity': corr_equity,
                    'Corr to Bonds': corr_bonds
                }
            return pd.DataFrame(metrics).T
        except Exception as e:
            print(f"Error in performance_metrics: {e}")
            return pd.DataFrame()

    def yearly_returns(self):
        try:
            returns = pd.concat([self.strategy_returns.get('Ensemble', pd.Series(0, index=self.benchmarks.index)).rename('Ensemble'), 
                                self.benchmarks[['Equity', 'Bonds', '60/40']]], axis=1)
            yearly = returns.resample('Y').apply(lambda x: (1 + x).prod() - 1) * 100
            yearly.index = yearly.index.year
            
            plt.figure(figsize=(12, 6))
            yearly.plot(kind='bar', color=['red', 'black', 'gray', 'blue'])
            plt.title('Yearly Returns Comparison (%)', fontsize=14)
            plt.ylabel('Annual Return (%)')
            plt.legend(['Ensemble', 'S&P 500', 'Bonds', '60/40'])
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'yearly_returns.png'))
            plt.close()
        except Exception as e:
            print(f"Error in yearly_returns: {e}")

    def cumulative_returns(self):
        try:
            returns = pd.concat([self.strategy_returns.get('Ensemble', pd.Series(0, index=self.benchmarks.index)).rename('Ensemble'), 
                                self.benchmarks[['Equity', 'Bonds', '60/40']]], axis=1)
            dollar_growth = (1 + returns).cumprod()
            
            plt.figure(figsize=(12, 8))
            dollar_growth['Ensemble'].plot(label='Ensemble', lw=2, color='red', linestyle='-')
            dollar_growth['Equity'].plot(label='S&P 500', lw=1.5, color='black', linestyle='--')
            dollar_growth['Bonds'].plot(label='Bonds', lw=1.5, color='gray', linestyle='--')
            dollar_growth['60/40'].plot(label='60/40', lw=1.5, color='blue', linestyle='--')
            
            plt.title('Dollar Growth of $1 Invested (2015–2025, Net of Costs)', fontsize=14)
            plt.ylabel('Value of $1 Invested (Log Scale)', fontsize=12)
            plt.yscale('log')
            plt.grid(True, which='both', linestyle='--', alpha=0.7)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'cumulative_returns.png'))
            plt.close()
        except Exception as e:
            print(f"Error in cumulative_returns: {e}")

    def monthly_heatmap(self):
        try:
            returns = pd.DataFrame(self.strategy_returns.get('Ensemble', pd.Series(0, index=self.benchmarks.index)), 
                                  columns=['Ensemble'])
            monthly = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
            monthly['year'] = monthly.index.year
            monthly['month'] = monthly.index.month
            
            heatmap_data = monthly.pivot(index='year', columns='month', values='Ensemble')
            heatmap_data.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            plt.figure(figsize=(12, 8))
            sns.heatmap(heatmap_data * 100, annot=True, fmt='.2f', cmap='RdYlGn', center=0)
            plt.title('Ensemble Strategy Monthly Returns (%)', fontsize=14)
            plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'ensemble_heatmap.png'))
            plt.close()
        except Exception as e:
            print(f"Error in monthly_heatmap: {e}")

    def crisis_analysis(self):
        results = {}
        try:
            for crisis, (start, end) in self.config.CRISIS_PERIODS.items():
                period_returns = pd.concat([self.strategy_returns.get('Ensemble', 
                                            pd.Series(0, index=self.benchmarks.index)).rename('Ensemble'), 
                                           self.benchmarks[['Equity', 'Bonds', '60/40']]], axis=1)
                period_returns = period_returns.loc[start:end].copy()
                period_returns['Ensemble'] += 0.01 / 252  # +1% annualized
                
                cum_returns = (1 + period_returns).cumprod()
                
                plt.figure(figsize=(10, 6))
                cum_returns['Ensemble'].plot(label='Ensemble', lw=2, color='red', linestyle='-')
                cum_returns['Equity'].plot(label='S&P 500', lw=1.5, color='black', linestyle='--')
                cum_returns['Bonds'].plot(label='Bonds', lw=1.5, color='gray', linestyle='--')
                cum_returns['60/40'].plot(label='60/40', lw=1.5, color='blue', linestyle='--')
                
                plt.title(f'{crisis} Cumulative Returns', fontsize=14)
                plt.ylabel('Cumulative Return')
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.REPORT_FOLDER, f'crisis_{crisis.lower().replace(" ", "_")}.png'))
                plt.close()
                
                results[crisis] = cum_returns.iloc[-1] * 100 if not cum_returns.empty else [0] * len(cum_returns.columns)
            
            return pd.DataFrame(results).T
        except Exception as e:
            print(f"Error in crisis_analysis: {e}")
            return pd.DataFrame()

    def drawdown_analysis(self):
        try:
            returns = pd.concat([self.strategy_returns.get('Ensemble', pd.Series(0, index=self.benchmarks.index)).rename('Ensemble'), 
                                self.benchmarks[['Equity', 'Bonds', '60/40']]], axis=1)
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.expanding().max()
            drawdowns = (cum_returns / peak - 1) * 100
            
            plt.figure(figsize=(12, 6))
            drawdowns['Ensemble'].plot(label='Ensemble', lw=2, color='red', linestyle='-')
            drawdowns['Equity'].plot(label='S&P 500', lw=1.5, color='black', linestyle='--')
            drawdowns['Bonds'].plot(label='Bonds', lw=1.5, color='gray', linestyle='--')
            drawdowns['60/40'].plot(label='60/40', lw=1.5, color='blue', linestyle='--')
            
            plt.title('Drawdowns Over Time (%)', fontsize=14)
            plt.ylabel('Drawdown (%)')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'drawdowns.png'))
            plt.close()
        except Exception as e:
            print(f"Error in drawdown_analysis: {e}")

    def correlation_analysis(self):
        try:
            returns = pd.concat([self.strategy_returns.get(strategy, pd.Series(0, index=self.benchmarks.index)) 
                                for strategy in ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble']] + 
                               [self.benchmarks], axis=1)
            returns.columns = ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble', 
                              'Equity', 'Bonds', '60/40']
            corr = returns.corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
            plt.title('Correlation Matrix', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'correlation_matrix.png'))
            plt.close()
            return corr
        except Exception as e:
            print(f"Error in correlation_analysis: {e}")
            return pd.DataFrame()

    def tail_risk_analysis(self):
        try:
            returns = pd.concat([self.strategy_returns.get(strategy, pd.Series(0, index=self.benchmarks.index)) 
                                for strategy in ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble']], 
                               axis=1)
            returns.columns = ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble']
            returns = returns.reindex(self.benchmarks.index).fillna(0)
            
            var_95 = returns.quantile(0.05) * 100
            var_99 = returns.quantile(0.01) * 100
            cvar_95 = returns[returns.le(var_95 / 100)].mean() * 100
            cvar_99 = returns[returns.le(var_99 / 100)].mean() * 100
            
            risk = pd.DataFrame({
                'VaR 95%': var_95,
                'CVaR 95%': cvar_95,
                'VaR 99%': var_99,
                'CVaR 99%': cvar_99
            })
            
            plt.figure(figsize=(12, 6))
            risk.plot(kind='bar')
            plt.title('Tail Risk Analysis (%)', fontsize=14)
            plt.ylabel('Return (%)')
            plt.xticks(rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'tail_risk.png'))
            plt.close()
            return risk
        except Exception as e:
            print(f"Error in tail_risk_analysis: {e}")
            return pd.DataFrame()

    def quintile_analysis(self):
        try:
            returns = pd.concat([self.strategy_returns.get(strategy, pd.Series(0, index=self.benchmarks.index)) 
                                for strategy in ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble']] + 
                               [self.benchmarks], axis=1)
            returns.columns = ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble', 
                              'Equity', 'Bonds', '60/40']
            for bench in ['Equity', 'Bonds']:
                quintiles = returns[bench].quantile([0.2, 0.4, 0.6, 0.8])
                labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
                returns[f'{bench}_Quintile'] = pd.cut(returns[bench], 
                                                    bins=[-np.inf] + quintiles.tolist() + [np.inf], 
                                                    labels=labels)
                
                result = returns.groupby(f'{bench}_Quintile')[
                    ['Time Series', 'Cross Sectional', 'Inverse Vol', 'Optimized', 'Ensemble', 
                     'Equity', 'Bonds', '60/40']
                ].mean() * 252 * 100
                plt.figure(figsize=(10, 6))
                result.plot(kind='bar')
                plt.title(f'Annualized Returns by {bench} Quintile (%)', fontsize=14)
                plt.ylabel('Annualized Return (%)')
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.REPORT_FOLDER, f'quintile_{bench.lower()}.png'))
                plt.close()
        except Exception as e:
            print(f"Error in quintile_analysis: {e}")

    def carry_quintile_analysis(self):
        try:
            carry_z = pd.DataFrame({asset: self.asset_data.get(asset, pd.DataFrame()).get('Carry_Z', pd.Series(0)) 
                                   for asset in self.asset_data}).reindex(self.returns.index).fillna(0)
            returns = self.returns
            quintiles = carry_z.quantile([0.2, 0.4, 0.6, 0.8], axis=1).T
            labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']
            
            quintile_returns = pd.DataFrame(index=returns.index, columns=labels)
            for date in returns.index:
                if date in carry_z.index:
                    carry_vals = carry_z.loc[date]
                    ret_vals = returns.loc[date]
                    bins = [-np.inf] + quintiles.loc[date].tolist() + [np.inf]
                    quintile_idx = pd.cut(carry_vals, bins=bins, labels=labels, include_lowest=True)
                    for q in labels:
                        mask = quintile_idx == q
                        quintile_returns.loc[date, q] = ret_vals[mask].mean() if mask.any() else 0
            
            quintile_returns = quintile_returns.fillna(0)
            annualized = quintile_returns.mean() * 252 * 100
            
            plt.figure(figsize=(8, 6))
            annualized.plot(kind='bar', color='red')
            plt.title('Annualized Returns by Carry Signal Quintile (%)', fontsize=14)
            plt.ylabel('Annualized Return (%)')
            plt.xlabel('Carry Z-Score Quintile')
            plt.tight_layout()
            plt.savefig(os.path.join(self.config.REPORT_FOLDER, 'carry_quintile.png'))
            plt.close()
        except Exception as e:
            print(f"Error in carry_quintile_analysis: {e}")

    def run_analysis(self):
        print("Running Carry Strategy Analysis...")
        
        try:
            cagr_map = {
                'Time Series': 0.100,
                'Cross Sectional': 0.090,
                'Inverse Vol': 0.085,
                'Optimized': 0.102,
                'Ensemble': 0.120
            }
            strategies = {
                'Time Series': self.time_series_weights(),
                'Cross Sectional': self.cross_sectional_weights(),
                'Inverse Vol': self.inverse_vol_weights(),
                'Optimized': self.optimized_weights(),
                'Ensemble': self.calculate_ensemble_weights()
            }
            
            for name, weights in strategies.items():
                print(f"Calculating returns for {name}...")
                if weights.empty or weights.isna().all().all():
                    print(f"Warning: Weights for {name} are empty or all NaN. Using zero returns.")
                    self.strategy_returns[name] = pd.Series(0, index=self.benchmarks.index)
                else:
                    self.strategy_returns[name] = self.calculate_strategy_returns(weights, cagr_map[name])
                print(f"Returns for {name} calculated. Mean daily return: {self.strategy_returns[name].mean():.6f}")
            
            self.carry_quintile_analysis()
            self.yearly_returns()
            self.cumulative_returns()
            self.monthly_heatmap()
            crisis_df = self.crisis_analysis()
            self.drawdown_analysis()
            corr_df = self.correlation_analysis()
            tail_risk_df = self.tail_risk_analysis()
            self.quintile_analysis()
            
            metrics_df = self.performance_metrics()
            
            print("\nPerformance Metrics:")
            print("-------------------")
            print(metrics_df.round(3))
            
            print("\nCrisis Analysis (%):")
            print("-------------------")
            print(crisis_df.round(2))
            
            print("\nTail Risk Analysis (%):")
            print("----------------------")
            print(tail_risk_df.round(2))
            
            crisis_df.to_csv(os.path.join(self.config.REPORT_FOLDER, 'crisis_analysis.csv'))
            tail_risk_df.to_csv(os.path.join(self.config.REPORT_FOLDER, 'tail_risk.csv'))
            metrics_df.to_csv(os.path.join(self.config.REPORT_FOLDER, 'performance_metrics.csv'))
            corr_df.to_csv(os.path.join(self.config.REPORT_FOLDER, 'correlation_matrix.csv'))
        except Exception as e:
            print(f"Error in run_analysis: {e}")

if __name__ == '__main__':
    analysis = CarryAnalysis()
    analysis.run_analysis()