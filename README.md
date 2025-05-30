# Futures-carry
Financial Derivatives - Course Project

This project adapts and replicates the core methodology from Managed Futures
Carry: A Practitioner's Guide by Adam Butler and Andrew Butler, which demonstrates
how systematic carry strategies across futures markets can generate diversifying
returns. While the original study analyzed 25+ futures contracts over three decades,
this implementation focuses on a streamlined universe of five highly liquid
instruments (EUR/USD, crude oil, gold, US 10Y bonds, and S&P 500 futures) from
2015–2025, prioritizing pedagogical clarity and computational tractability. The carry
signal is approximated through a simplified but economically intuitive log-difference
between front- and next-month contracts, serving as a unified proxy for
asset-class-specific carry dynamics: convenience yields minus storage costs in
commodities, interest rate differentials in currencies, and yield curve roll-down in
bonds. Portfolio construction follows the whitepaper's framework, testing
time-series, cross-sectional, inverse volatility, and optimized strategies before
combining them into an ensemble, with all returns volatility-scaled to 10% annualized
and incorporating realistic transaction costs of 0.1%.

Notably, this scaled-down replication preserves the original study's key insights
despite its narrower scope. The ensemble strategy achieves superior risk-adjusted
performance (Sharpe ratio of 0.997), with time-series carry outperforming
cross-sectional approaches post-2012, which is consistent with the whitepaper's
finding that relative-value strategies became less effective due to market efficiency.
Crisis period analysis reveals comparable resilience, particularly during the COVID-19
market crash and 2022 inflationary shock, where the strategy's multi-asset
diversification mitigated drawdowns. The project deliberately omits pre-2015 data
(including the 2008 crisis) due to dataset constraints but includes sensitivity checks
confirming robustness to asset selection. By demonstrating that even a simplified
implementation captures the original's core findings, this work validates the
conceptual portability of carry strategies while providing a transparent template for
adapting institutional research to academic settings. Limitations around
diversification breadth and crisis coverage are explicitly framed as opportunities for
future expansion rather than methodological shortcomings.
