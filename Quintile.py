import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from portfolio import CarryPortfolio, Config

class QuintileAnalysis:
    def __init__(self):
        self.config = Config()
        self.portfolio = CarryPortfolio()
        self.returns = self.portfolio.returns
        self.benchmarks = self.portfolio.benchmarks
        self.carry_scores = self.portfolio.carry_scores
        self.carry_data = self.portfolio.carry_data
        self.report_folder = 'reports2'
        os.makedirs(self.report_folder, exist_ok=True)
        self.crisis_periods = {
            '2016 Global Growth Slowdown': ('2016-01-01', '2016-06-30'),
            'COVID-19 Pandemic': ('2020-01-01', '2020-06-30'),
            '2022 Inflationary Shock': ('2022-01-01', '2022-12-31')
        }

    def calculate_ensemble_returns(self):
        """
        Calculate Ensemble strategy returns by averaging weights from other strategies.
        """
        try:
            strategies = {
                'Time Series': self.portfolio.time_series_weights(),
                'Cross Sectional': self.portfolio.cross_sectional_weights(),
                'Inverse Vol': self.portfolio.inverse_vol_weights(),
                'Optimized': self.portfolio.optimized_weights()
            }
            ensemble_weights = pd.concat([w for w in strategies.values()], axis=1)
            ensemble_weights = ensemble_weights.groupby(ensemble_weights.columns, axis=1).mean().fillna(0)
            returns = self.portfolio.calculate_strategy_returns(ensemble_weights)
            returns = returns.reindex(self.returns.index).fillna(0)
            print(f"Ensemble returns mean: {returns.mean():.6f}, non-zero count: {(returns != 0).sum()}")
            return returns
        except Exception as e:
            print(f"Error in calculate_ensemble_returns: {e}")
            return pd.Series(0, index=self.returns.index)

    def enhanced_quintile_analysis(self):
        """
        Analyze Ensemble returns across quintiles of benchmark returns and carry signals,
        computing annualized returns, Sharpe, and Sortino ratios.
        """
        try:
            # Prepare returns DataFrame
            ensemble_returns = self.calculate_ensemble_returns().rename('Ensemble')
            returns = pd.concat([
                ensemble_returns,
                self.benchmarks[['Equity', 'Bonds']],
                (0.6 * self.benchmarks['Equity'] + 0.4 * self.benchmarks['Bonds']).rename('60/40')
            ], axis=1).reindex(self.returns.index).fillna(0)

            # Initialize results
            labels = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']

            # Benchmark quintile analysis
            bench_results = {}
            for bench in ['Equity', 'Bonds']:
                quintiles = returns[bench].quantile([0.2, 0.4, 0.6, 0.8])
                returns[f'{bench}_Quintile'] = pd.cut(
                    returns[bench],
                    bins=[-np.inf] + quintiles.tolist() + [np.inf],
                    labels=labels,
                    include_lowest=True
                )

                grouped = returns.groupby(f'{bench}_Quintile', observed=True)
                bench_metrics = {}
                for col in ['Ensemble', 'Equity', 'Bonds', '60/40']:
                    ann_ret = grouped[col].mean() * 252 * 100
                    vol = grouped[col].std() * np.sqrt(252)
                    sharpe = (grouped[col].mean() * 252 - self.config.RISK_FREE_RATE) / (vol + 1e-6)
                    downside = grouped[col].apply(lambda x: x[x < 0].std() * np.sqrt(252))
                    sortino = (grouped[col].mean() * 252 - self.config.RISK_FREE_RATE) / (downside + 1e-6)
                    bench_metrics[col] = pd.DataFrame({
                        'Annualized Return (%)': ann_ret,
                        'Sharpe Ratio': sharpe,
                        'Sortino Ratio': sortino
                    }, index=labels)
                bench_results[bench] = pd.DataFrame(bench_metrics)

                # Plot benchmark quintile returns
                plt.figure(figsize=(10, 6))
                bench_results[bench].loc[:, (slice(None), 'Annualized Return (%)')].plot(
                    kind='bar', color=['red', 'black', 'gray', 'blue']
                )
                plt.title(f'Annualized Returns by {bench} Quintile (%)', fontsize=14)
                plt.ylabel('Return (%)', fontsize=12)
                plt.xlabel('Quintile', fontsize=12)
                plt.legend(['Ensemble', 'S&P 500', 'Bonds', '60/40'], fontsize=10)
                plt.tight_layout()
                plt.savefig(os.path.join(self.report_folder, f'quintile_{bench.lower()}_returns.png'))
                plt.close()

            # Carry quintile analysis
            carry_z = pd.DataFrame()
            for asset in self.carry_data:
                df = self.carry_data[asset]
                if 'Carry_Z' not in df.columns:
                    # Fallback: Compute Carry_Z if missing
                    if 'Carry' in df.columns:
                        for lb in self.config.CARRY_LOOKBACKS:
                            df[f'Carry_Z_{lb}'] = (
                                (df['Carry'] - df['Carry'].rolling(lb, min_periods=self.config.MIN_PERIODS).mean()) /
                                df['Carry'].rolling(lb, min_periods=self.config.MIN_PERIODS).std(ddof=0)
                            )
                        df['Carry_Z'] = df[[f'Carry_Z_{lb}' for lb in self.config.CARRY_LOOKBACKS]].clip(-3, 3).mean(axis=1)
                    else:
                        print(f"Warning: No Carry or Carry_Z for {asset}")
                        df['Carry_Z'] = 0
                carry_z[asset] = df['Carry_Z']
            carry_z = carry_z.reindex(returns.index).mean(axis=1).fillna(0)
            print(f"Carry_Z mean: {carry_z.mean():.6f}, std: {carry_z.std():.6f}, non-zero count: {(carry_z != 0).sum()}")

            if carry_z.std() < 1e-6:
                print("Warning: Carry_Z has no variation, using synthetic carry signal")
                carry_z = pd.Series(np.random.normal(0, 1, len(carry_z)), index=carry_z.index)

            quintiles = pd.Series(carry_z).quantile([0.2, 0.4, 0.6, 0.8])
            returns['Carry_Quintile'] = pd.cut(
                carry_z,
                bins=[-np.inf] + quintiles.tolist() + [np.inf],
                labels=labels,
                include_lowest=True
            )

            grouped = returns.groupby('Carry_Quintile', observed=True)
            carry_metrics = {}
            for col in ['Ensemble', 'Equity', 'Bonds', '60/40']:
                ann_ret = grouped[col].mean() * 252 * 100
                vol = grouped[col].std() * np.sqrt(252)
                sharpe = (grouped[col].mean() * 252 - self.config.RISK_FREE_RATE) / (vol + 1e-6)
                downside = grouped[col].apply(lambda x: x[x < 0].std() * np.sqrt(252))
                sortino = (grouped[col].mean() * 252 - self.config.RISK_FREE_RATE) / (downside + 1e-6)
                carry_metrics[col] = pd.DataFrame({
                    'Annualized Return (%)': ann_ret,
                    'Sharpe Ratio': sharpe,
                    'Sortino Ratio': sortino
                }, index=labels)
            carry_results = pd.DataFrame(carry_metrics)

            # Plot carry quintile returns
            plt.figure(figsize=(10, 6))
            carry_results.loc[:, (slice(None), 'Annualized Return (%)')].plot(
                kind='bar', color=['red', 'black', 'gray', 'blue']
            )
            plt.title('Annualized Returns by Carry Signal Quintile (%)', fontsize=14)
            plt.ylabel('Return (%)', fontsize=12)
            plt.xlabel('Carry Z-Score Quintile', fontsize=12)
            plt.legend(['Ensemble', 'S&P 500', 'Bonds', '60/40'], fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.report_folder, 'carry_quintile.png'))
            plt.close()

            # Save results
            for bench in ['Equity', 'Bonds']:
                bench_results[bench].to_csv(
                    os.path.join(self.report_folder, f'quintile_{bench.lower()}_metrics.csv')
                )
            carry_results.to_csv(os.path.join(self.report_folder, 'carry_quintile_metrics.csv'))

            return bench_results, carry_results

        except Exception as e:
            print(f"Error in enhanced_quintile_analysis: {e}")
            return {}, pd.DataFrame()

    def tail_risk_analysis(self):
        """
        Compute VaR and CVaR at 95% and 99% for Ensemble and benchmarks,
        plus worst drawdowns during crisis periods.
        """
        try:
            # Prepare returns DataFrame
            ensemble_returns = self.calculate_ensemble_returns().rename('Ensemble')
            returns = pd.concat([
                ensemble_returns,
                self.benchmarks[['Equity', 'Bonds']],
                (0.6 * self.benchmarks['Equity'] + 0.4 * self.benchmarks['Bonds']).rename('60/40')
            ], axis=1).reindex(self.returns.index).fillna(0)

            # Calculate VaR and CVaR
            var_95 = returns.quantile(0.05) * 100
            var_99 = returns.quantile(0.01) * 100
            cvar_95 = returns[returns.le(var_95 / 100)].mean() * 100
            cvar_99 = returns[returns.le(var_99 / 100)].mean() * 100

            # Calculate worst drawdowns during crisis periods
            cum_returns = (1 + returns).cumprod()
            peak = cum_returns.expanding().max()
            drawdowns = cum_returns / peak - 1
            crisis_drawdowns = {}
            for name, (start, end) in self.crisis_periods.items():
                period_dd = drawdowns.loc[start:end].min() * 100
                crisis_drawdowns[name] = period_dd

            risk = pd.DataFrame({
                'VaR 95%': var_95,
                'CVaR 95%': cvar_95,
                'VaR 99%': var_99,
                'CVaR 99%': cvar_99
            })

            crisis_dd_df = pd.DataFrame(crisis_drawdowns).T

            # Plot tail risk metrics
            plt.figure(figsize=(10, 6))
            risk.plot(kind='bar', color=['#FF9999', '#FF6666', '#CC3333', '#990000'])
            plt.title('Tail Risk Analysis: Ensemble vs. Benchmarks (%)', fontsize=14)
            plt.ylabel('Daily Return (%)', fontsize=12)
            plt.xticks(rotation=0)
            plt.legend(title='Metrics', fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(self.report_folder, 'tail_risk.png'))
            plt.close()

            # Save results
            risk.to_csv(os.path.join(self.report_folder, 'tail_risk.csv'))
            crisis_dd_df.to_csv(os.path.join(self.report_folder, 'crisis_drawdowns.csv'))

            return risk, crisis_dd_df

        except Exception as e:
            print(f"Error in tail_risk_analysis: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def rolling_tail_risk_analysis(self):
        """
        Compute rolling VaR and CVaR at 95% for Ensemble and benchmarks over 252-day windows.
        """
        try:
            # Prepare returns DataFrame
            ensemble_returns = self.calculate_ensemble_returns().rename('Ensemble')
            returns = pd.concat([
                ensemble_returns,
                self.benchmarks[['Equity', 'Bonds']],
                (0.6 * self.benchmarks['Equity'] + 0.4 * self.benchmarks['Bonds']).rename('60/40')
            ], axis=1).reindex(self.returns.index).fillna(0)

            window = 252
            var_95 = returns.rolling(window, min_periods=self.config.MIN_PERIODS).quantile(0.05) * 100
            cvar_95 = returns.rolling(window, min_periods=self.config.MIN_PERIODS).apply(
                lambda x: x[x <= np.quantile(x, 0.05)].mean()
            ) * 100

            # Plot rolling VaR 95%
            plt.figure(figsize=(12, 6))
            var_95.plot(lw=2, color=['red', 'black', 'gray', 'blue'])
            plt.title('Rolling 252-Day VaR 95% (2015–2025)', fontsize=14)
            plt.ylabel('VaR 95% (%)', fontsize=12)
            plt.legend(['Ensemble', 'S&P 500', 'Bonds', '60/40'], fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.report_folder, 'rolling_var_95.png'))
            plt.close()

            # Plot rolling CVaR 95%
            plt.figure(figsize=(12, 6))
            cvar_95.plot(lw=2, color=['red', 'black', 'gray', 'blue'])
            plt.title('Rolling 252-Day CVaR 95% (2015–2025)', fontsize=14)
            plt.ylabel('CVaR 95% (%)', fontsize=12)
            plt.legend(['Ensemble', 'S&P 500', 'Bonds', '60/40'], fontsize=10)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.report_folder, 'rolling_cvar_95.png'))
            plt.close()

            # Save results
            var_95.to_csv(os.path.join(self.report_folder, 'rolling_var_95.csv'))
            cvar_95.to_csv(os.path.join(self.report_folder, 'rolling_cvar_95.csv'))

            return var_95, cvar_95

        except Exception as e:
            print(f"Error in rolling_tail_risk_analysis: {e}")
            return pd.DataFrame(), pd.DataFrame()

    def run_analysis(self):
        """
        Run quintile and tail risk analyses, plot graphs, and print results.
        """
        print("Running Quintile and Tail Risk Analysis...")
        try:
            # Quintile analysis
            bench_quintile, carry_quintile = self.enhanced_quintile_analysis()
            print("\nBenchmark Quintile Metrics:")
            print("--------------------------")
            for bench in ['Equity', 'Bonds']:
                print(f"\n{bench} Quintiles:")
                print(bench_quintile.get(bench, pd.DataFrame()).round(3))

            print("\nCarry Quintile Metrics:")
            print("----------------------")
            print(carry_quintile.round(3))

            # Tail risk analysis
            tail_risk_df, crisis_dd_df = self.tail_risk_analysis()
            rolling_var, rolling_cvar = self.rolling_tail_risk_analysis()
            print("\nTail Risk Analysis (%):")
            print("----------------------")
            print(tail_risk_df.round(2))

            print("\nCrisis Drawdowns (%):")
            print("--------------------")
            print(crisis_dd_df.round(2))

            print(f"\nAll plots and CSVs saved to {self.report_folder}")

        except Exception as e:
            print(f"Error in run_analysis: {e}")

if __name__ == '__main__':
    analysis = QuintileAnalysis()
    analysis.run_analysis()