# Integrated Rolling Simulation Report

## 1. Executive Summary
- Run ID: 20260514_195851
- Figures cleaned before run: True
- Run-specific archive: /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/runs/20260514_195851
- Pair IDs: [1]
- Truncated debug run: False
- Row-level trade files saved: False
- Rows: 846332
- Stocks: 20
- Dates: 19
- Stock-days: 380
- Daily Sharpe meaningful: YES
- Integration mechanics: PASS
- Sizing plausibility: PASS
- PnL formula checks: PASS
- Fitted evaluator checks: PASS

## 2. Data and Sample
- Train/test months and stock universes are read from teammate rolling_pair_summary.csv.
- Row-level trade files were not saved. Portfolio path reconciliation checks that require row-level trades are skipped, while summary-level metrics are still reported.
 pair_id  n_rows  n_stocks  number_of_unique_dates  total_gross_pnl  total_net_pnl  total_signed_impact_cost_normalized  total_quadratic_impact_cost_normalized  mean_daily_net_pnl  std_daily_net_pnl  daily_sharpe  annualized_sharpe  total_signed_volume_turnover  total_notional_turnover  average_daily_turnover_shares  average_daily_turnover_notional  total_normalized_turnover  max_participation_rate  mean_participation_rate  max_drawdown  max_daily_drawdown  max_abs_position  max_abs_impact  number_of_trades  number_of_stock_days  mean_abs_trade  median_abs_trade   rows  rows_with_scaling  rows_skipped_due_to_missing_scaling
       1  846332        20                      19     2.217032e+07   2.217032e+07                             0.000002                                0.000005        1.166859e+06      453939.882354      2.570514          40.805642                  6.447507e+09             4.561109e+11                   3.393425e+08                     2.400584e+10                     2.3712                    0.01                 0.006617 -5.681374e+06                 0.0     462046.461905        0.000055            669454                   380     7618.177272       2130.778437 846332             846332                                    0
- Daily Sharpe is meaningful for this run.

## 3. Alpha Diagnostics
- Baseline synthetic alpha: h=5m, rho=0.10, H_alpha=5m unless configured otherwise.
- alpha_mean_bps: None
- alpha_std_bps: None
- corr(alpha, future_return_h): None
- corr(position_before, future_return_h): None
- corr(trade, alpha): None
- corr(position_after, alpha): None
- share_sign_position_matches_alpha: None
- share_sign_trade_matches_alpha: None
- gross alpha capture: None
- Interpretation: The strategy captures alpha before fitted impact costs.

## 4. Strategy Sizing Diagnostics
- mean_abs_trade: None
- median_abs_trade: None
- max_abs_trade: None
- max_abs_position: None
- total_signed_volume_turnover: None
- total_notional_turnover: None
- max_participation_rate: None
- mean_participation_rate: None
- share_trade_clipped: None
- share_position_clipped: None
- max_position_over_ADV: None
- Previous unconstrained runs were unrealistic. The capped runs are the reportable ones.

## 5. OW Internal Wealth
- total_gross_pnl: 2.2170e+07
- total_internal_impact_cost_normalized: nan
- total_net_pnl_internal: nan
- internal_impact_cost_to_gross_pnl_ratio: nan
- max_drawdown: -5.6814e+06
- Internal OW costs are normalized diagnostics and are not on the same economic scale as fitted-regression price-unit costs.
- Internal cost scale warning not triggered.

## 6. Fitted Regression Evaluator Wealth
- total_fitted_cost_ow_regression: 7.6224e+07
- total_abs_fitted_cost_ow_regression: 7.7272e+07
- net_pnl_under_ow_regression_eval: -5.4054e+07
- mean_abs_marginal_impact_ow_regression_bps: 0.8897
- total_fitted_cost_reduced_form: 7.4695e+07
- total_abs_fitted_cost_reduced_form: 7.7084e+07
- net_pnl_under_reduced_form_eval: -5.2524e+07
- mean_abs_marginal_impact_reduced_form_bps: 0.8907
No wrong-model summary.
- Interpretation: The alpha strategy is profitable before fitted impact costs, but fitted regression impact costs more than offset gross PnL on this sample.
- Model comparison: Reduced-form evaluator is less punitive than OW_transient on this sample.

## 7. Fitted-Regression Proxy Strategy
- Because teammate's fitted reduced_form model is an OLS regression, not a structural impact dynamics, it does not provide a closed-form dynamic optimal strategy.
- I therefore implement a local myopic quadratic-cost proxy using the regression marginal slope with respect to strategy order flow.
- The proxy uses orderFlow_scenario = orderFlow_market + q_strategy and estimates d(ret_bps)/dq from x_flow, x_flow_depth, and optionally x_trade.
- Objective: maximize q * alpha_price - impact_slope_price_per_share * q^2 - inventory_penalty * (Q_prev + q)^2.
- This is reportable as a fitted-regression-aware proxy strategy, not as the course structural AFS optimum.
- total_gross_pnl_fitted_proxy: None
- total_fitted_proxy_cost: None
- total_net_pnl_fitted_proxy: None
- total_turnover_fitted_proxy: None
- max_drawdown_fitted_proxy: None
- max_participation_rate_fitted_proxy: None
- mean_participation_rate_fitted_proxy: None
- rows in first pair proxy trade table: 0

## 8. PnL Formula and Timing Validation
                                            check status                                                                                                                                                                                                                                                       message  pair_id
                            delta_mid_consistency   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
                   gross_pnl_uses_position_before   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
                                position_dynamics   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
                             daily_position_reset   PASS                                                                                                                                                                                                                                      max_first_abs_position=0      1.0
                                   final_position   WARN                                                                                                                                                                                                                   max_final_abs_position=3.7e+05; capped=True      1.0
                            no_lookahead_pnl_note   PASS                                                                                                                                                            gross PnL is position_before*delta_mid; aggregate difference vs position_after convention=7.36e+07      1.0
                          alpha_trade_timing_note   PASS                                                                                                                                                            trade at row t uses alpha_t and affects position_after_t; PnL over t-1 to t uses position_before_t      1.0
                         alpha_future_return_corr   PASS                                                                                                                                                                                                                                                   corr=0.1904      1.0
                      position_future_return_corr   WARN                                                                                                                                                                                                                                                 corr=-0.03486      1.0
                                 trade_alpha_corr   PASS                                                                                                                                                                                                        corr_trade_alpha=0.005467; corr_position_alpha=0.03807      1.0
                                   sign_alignment   PASS                                                                                                                                                                               share_position_same_sign_as_alpha=59.22%; share_trade_same_sign_as_alpha=53.06%      1.0
                              gross_alpha_capture   PASS                                                                                                                                                                                                                                         gross_pnl=2.21703e+07      1.0
     ow_regression_orderflow_market_plus_strategy   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
            ow_regression_marginal_impact_formula   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
               ow_regression_bps_to_price_formula   PASS                                                                                                                                                                                                                                            max_error=5.55e-17      1.0
                       ow_regression_cost_formula   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
                   ow_regression_cost_sign_sanity   PASS                                                                                                                                  share_same_direction=92.16%; share_positive_cost=92.16%; total_signed=7.62243e+07; total_abs=7.72723e+07; netout_ratio=0.986      1.0
        ow_regression_cost_subtraction_convention   PASS                                                                                                                                                  net_pnl_fitted_model = gross_pnl - fitted_impact_cost_signed; positive fitted cost is adverse and subtracted      1.0
      reduced_form_orderflow_market_plus_strategy   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
             reduced_form_marginal_impact_formula   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
                reduced_form_bps_to_price_formula   PASS                                                                                                                                                                                                                                            max_error=4.44e-16      1.0
                        reduced_form_cost_formula   PASS                                                                                                                                                                                                                                                   max_error=0      1.0
                    reduced_form_cost_sign_sanity   PASS                                                                                                                                  share_same_direction=90.58%; share_positive_cost=90.58%; total_signed=7.46947e+07; total_abs=7.70841e+07; netout_ratio=0.969      1.0
         reduced_form_cost_subtraction_convention   PASS                                                                                                                                                  net_pnl_fitted_model = gross_pnl - fitted_impact_cost_signed; positive fitted cost is adverse and subtracted      1.0
                                participation_cap   PASS                                                                                                                                                                                                                           max=0.01; mean=0.00661692; cap=0.01      1.0
                                position_over_adv   PASS                                                                                                                                                                                                                                                      max=0.05      1.0
                              trade_clipped_share   WARN                                                                                                                                                                                                                                                  share=82.50%      1.0
                     fitted_marginal_impact_scale   PASS                                                                                                                                                                                                 mean_abs=0.8907; median_abs=0.558; p95_abs=2.983; max_abs=261      1.0
                                notional_turnover   PASS                                                                                                                                                                                                                           total_notional_turnover=4.56111e+11      1.0
                   all_pairs_strategy_summary.csv   PASS                                                                                         /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/all_pairs_strategy_summary.csv      NaN
                     all_pairs_stress_summary.csv   PASS                                                                                           /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/all_pairs_stress_summary.csv      NaN
                     all_selected_pairs_processed   PASS                                                                                                                                                                                                                                                    missing=[]      NaN
                                    wealth_finite   PASS                                                                                                                                                                                                                                          aggregate pnl finite      NaN
                      participation_rate_reported   PASS                                                                                                                                                                                                                                   max_participation_rate=0.01      NaN
                 total_notional_turnover_reported   PASS                                                                                                                                                                                                                           total_notional_turnover=4.56111e+11      NaN
                 figure_total_net_pnl_by_pair.png   PASS                         path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/figures/total_net_pnl_by_pair.png; exists=True; size=16405; modified_after_run_start=True      NaN
          figure_pair_cumulative_wealth_my_ow.png   PASS           path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/pair_cumulative_wealth_my_ow.png; exists=True; size=38973; modified_after_run_start=True      NaN
          figure_pair_fitted_evaluator_wealth.png   PASS           path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/pair_fitted_evaluator_wealth.png; exists=True; size=46761; modified_after_run_start=True      NaN
  figure_cumulative_wealth_internal_vs_fitted.png   PASS   path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/cumulative_wealth_internal_vs_fitted.png; exists=True; size=54288; modified_after_run_start=True      NaN
   figure_gross_pnl_vs_fitted_cost_cumulative.png   PASS    path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/gross_pnl_vs_fitted_cost_cumulative.png; exists=True; size=56516; modified_after_run_start=True      NaN
         figure_marginal_impact_bps_histogram.png   PASS          path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/marginal_impact_bps_histogram.png; exists=True; size=24811; modified_after_run_start=True      NaN
          figure_participation_rate_histogram.png   PASS           path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/participation_rate_histogram.png; exists=True; size=26677; modified_after_run_start=True      NaN
           figure_position_over_ADV_histogram.png   PASS            path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/position_over_ADV_histogram.png; exists=True; size=27053; modified_after_run_start=True      NaN
      figure_forced_liq_hard_vs_capped_wealth.png   PASS       path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_hard_vs_capped_wealth.png; exists=True; size=52299; modified_after_run_start=True      NaN
       figure_forced_liq_event_costs_by_stock.png   PASS        path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_event_costs_by_stock.png; exists=True; size=42019; modified_after_run_start=True      NaN
figure_forced_liq_residual_inventory_by_stock.png   PASS path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_residual_inventory_by_stock.png; exists=True; size=60482; modified_after_run_start=True      NaN
          figure_forced_liq_time_to_liquidate.png   PASS           path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_time_to_liquidate.png; exists=True; size=20898; modified_after_run_start=True      NaN
        figure_forced_liq_participation_rates.png   PASS         path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_participation_rates.png; exists=True; size=25800; modified_after_run_start=True      NaN
                figure_forced_liq_sample_path.png   PASS                 path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_sample_path.png; exists=True; size=68660; modified_after_run_start=True      NaN
             portfolio_internal_unique_timestamps   SKIP                                                                                                                                                                                        row-level internal trades not saved; summary metrics validated instead      1.0
   portfolio_internal_final_matches_total_net_pnl   SKIP                                                                                                                                                                                                                           row-level internal trades not saved      1.0
                        portfolio_plot_data_build   SKIP                                                                                                                                                                                                  row-level trade/evaluator files not saved for reconciliation      1.0
                           hard_block_trades_file   PASS                                                                         /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/stress/forced_liq_hard_block_trades.csv      1.0
                           hard_block_events_file   PASS                                                                         /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/stress/forced_liq_hard_block_events.csv      1.0
                 hard_block_fitted_evaluator_file   PASS                                                               /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/stress/forced_liq_hard_block_fitted_evaluator.csv      1.0
                            hard_block_final_flat   PASS                                                                                                                                                                                                                                  max_abs_position_after_liq=0      1.0
                hard_block_cap_violation_reported   PASS                                                                                                                                                                                          cap_violation_rate=85.52%; violation is allowed in hard-block stress      1.0
                  fitted_liquidation_cost_nonzero   PASS                                                                                                                                                                                                                                         zero_cost_share=0.54%      1.0
                                      event_count   PASS                                                                                                                                                                                                                                    events=380; stock_days=380      1.0
                      capped_residual_trades_file   PASS                                                                    /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/stress/forced_liq_capped_residual_trades.csv      1.0
                      capped_residual_events_file   PASS                                                                    /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/stress/forced_liq_capped_residual_events.csv      1.0
            capped_residual_fitted_evaluator_file   PASS                                                          /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/stress/forced_liq_capped_residual_fitted_evaluator.csv      1.0
                    capped_residual_cap_respected   PASS                                                                                                                                                                                                                          max_liq_participation=0.01; cap=0.01      1.0
                        capped_residual_direction   PASS                                                                                                                                                                                                         liquidation trades reduce absolute residual inventory      1.0
                             liquidation_priority   PASS                                                                                                                                                                                                                                alpha_trades_suppressed=425534      1.0
                    residual_inventory_accounting   PASS                                                                                                                                                                                                                          total_abs_final_residual=3.98813e-10      1.0
                  fitted_liquidation_cost_nonzero   PASS                                                                                                                                                                                                                                         zero_cost_share=0.00%      1.0
                                      event_count   PASS                                                                                                                                                                                                                                    events=380; stock_days=380      1.0

## 9. Stress Tests and Sensitivity
 pair_id                      scenario_name             scenario_type  total_gross_pnl  total_net_pnl  total_signed_impact_cost_normalized  total_quadratic_impact_cost_normalized  mean_daily_net_pnl  std_daily_net_pnl  daily_sharpe  annualized_sharpe  total_signed_volume_turnover  total_notional_turnover  average_daily_turnover_shares  average_daily_turnover_notional  total_normalized_turnover  max_participation_rate  mean_participation_rate  max_drawdown  max_daily_drawdown  max_abs_position  max_abs_impact  number_of_trades  number_of_stock_days  mean_abs_trade  median_abs_trade   rows  rows_with_scaling  rows_skipped_due_to_missing_scaling  total_fitted_cost_ow_regression  total_abs_fitted_cost_ow_regression  net_pnl_under_ow_regression_eval  mean_abs_marginal_impact_ow_regression_bps  total_fitted_cost_reduced_form  total_abs_fitted_cost_reduced_form  net_pnl_under_reduced_form_eval  mean_abs_marginal_impact_reduced_form_bps     liquidation_mode  number_of_liquidation_events  number_fully_liquidated_immediately  number_fully_liquidated_eventually  number_with_residual_after_first_liquidation  number_with_overnight_residual  max_residual_position  total_abs_residual_position  mean_time_to_liquidate_minutes  max_time_to_liquidate_minutes  total_alpha_trades_suppressed_due_to_liquidation  hard_block_cap_violation_rate
       1      forced_liq_hard_block_stop_OW forced_liquidation_stress     1.448707e+07   1.448707e+07                                  0.0                                     0.0       762477.527526      394716.270765      1.931710          30.664952                  2.425080e+09             1.699190e+11                   1.276358e+08                     8.943108e+09                     2.3712                    0.05                 0.002472 -3.843124e+06                 0.0     462046.461905        0.000055            244293                   380     2865.400738               0.0 846332             846332                                    0                     2.906035e+07                         2.960862e+07                     -1.457327e+07                                    0.496215                    3.014232e+07                        3.104972e+07                    -1.565525e+07                                   0.528655           hard_block                           380                                  380                                 380                                             0                               0           0.000000e+00                 0.000000e+00                        0.000000                            0.0                                            425230                       0.855228
       1 forced_liq_capped_residual_stop_OW forced_liquidation_stress     1.447306e+07   1.447306e+07                                  0.0                                     0.0       761740.193304      395383.170160      1.926587          30.583625                  2.425080e+09             1.699191e+11                   1.276358e+08                     8.943111e+09                     2.3712                    0.01                 0.002472 -3.859668e+06                 0.0     462046.461905        0.000055            245198                   380     2865.400738               0.0 846332             846332                                    0                     2.850567e+07                         2.905591e+07                     -1.403261e+07                                    0.496008                    2.953005e+07                        3.045758e+07                    -1.505699e+07                                   0.528452 capped_with_residual                           380                                   61                                 380                                           319                               0           6.548362e-11                 3.988134e-10                        0.426316                            1.5                                            425534                       0.000000

## 9a. Forced Liquidation Stress: Hard Block vs Capped Residual
- Hard block liquidation forces Q -> 0 immediately at the liquidation timestamp and may violate participation caps by design.
- Capped residual liquidation respects liquidation participation caps. If the position cannot be fully liquidated, residual inventory remains and liquidation trades have priority over new alpha trades until cleared.
- Hard block answers the exact Section 2.7 single-block stress question; capped residual is the operationally realistic variant with inventory carry risk.
 pair_id                      scenario_name             scenario_type  total_gross_pnl  total_net_pnl  total_signed_impact_cost_normalized  total_quadratic_impact_cost_normalized  mean_daily_net_pnl  std_daily_net_pnl  daily_sharpe  annualized_sharpe  total_signed_volume_turnover  total_notional_turnover  average_daily_turnover_shares  average_daily_turnover_notional  total_normalized_turnover  max_participation_rate  mean_participation_rate  max_drawdown  max_daily_drawdown  max_abs_position  max_abs_impact  number_of_trades  number_of_stock_days  mean_abs_trade  median_abs_trade   rows  rows_with_scaling  rows_skipped_due_to_missing_scaling  total_fitted_cost_ow_regression  total_abs_fitted_cost_ow_regression  net_pnl_under_ow_regression_eval  mean_abs_marginal_impact_ow_regression_bps  total_fitted_cost_reduced_form  total_abs_fitted_cost_reduced_form  net_pnl_under_reduced_form_eval  mean_abs_marginal_impact_reduced_form_bps     liquidation_mode  number_of_liquidation_events  number_fully_liquidated_immediately  number_fully_liquidated_eventually  number_with_residual_after_first_liquidation  number_with_overnight_residual  max_residual_position  total_abs_residual_position  mean_time_to_liquidate_minutes  max_time_to_liquidate_minutes  total_alpha_trades_suppressed_due_to_liquidation  hard_block_cap_violation_rate
       1      forced_liq_hard_block_stop_OW forced_liquidation_stress     1.448707e+07   1.448707e+07                                  0.0                                     0.0       762477.527526      394716.270765      1.931710          30.664952                  2.425080e+09             1.699190e+11                   1.276358e+08                     8.943108e+09                     2.3712                    0.05                 0.002472 -3.843124e+06                 0.0     462046.461905        0.000055            244293                   380     2865.400738               0.0 846332             846332                                    0                     2.906035e+07                         2.960862e+07                     -1.457327e+07                                    0.496215                    3.014232e+07                        3.104972e+07                    -1.565525e+07                                   0.528655           hard_block                           380                                  380                                 380                                             0                               0           0.000000e+00                 0.000000e+00                        0.000000                            0.0                                            425230                       0.855228
       1 forced_liq_capped_residual_stop_OW forced_liquidation_stress     1.447306e+07   1.447306e+07                                  0.0                                     0.0       761740.193304      395383.170160      1.926587          30.583625                  2.425080e+09             1.699191e+11                   1.276358e+08                     8.943111e+09                     2.3712                    0.01                 0.002472 -3.859668e+06                 0.0     462046.461905        0.000055            245198                   380     2865.400738               0.0 846332             846332                                    0                     2.850567e+07                         2.905591e+07                     -1.403261e+07                                    0.496008                    2.953005e+07                        3.045758e+07                    -1.505699e+07                                   0.528452 capped_with_residual                           380                                   61                                 380                                           319                               0           6.548362e-11                 3.988134e-10                        0.426316                            1.5                                            425534                       0.000000

No sensitivity scenarios saved.

## 10. Figures
- Portfolio-level plots aggregate PnL and costs across all stocks at each timestamp before taking cumulative sums.
- Sample path plots explicitly use one selected stock/day only.
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/alpha_position_alignment_sample.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/clipped_trade_share_by_day.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/cumulative_wealth_internal_vs_fitted.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/drawdown_internal_vs_fitted.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/fitted_cost_vs_gross_pnl_scatter.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_event_costs_by_stock.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_hard_vs_capped_wealth.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_participation_rates.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_residual_inventory_by_stock.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_sample_path.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/forced_liq_time_to_liquidate.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/gross_pnl_vs_fitted_cost_cumulative.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/marginal_impact_bps_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/marginal_impact_bps_timeseries_sample.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/pair_cumulative_wealth_my_ow.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/pair_fitted_evaluator_wealth.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/pair_marginal_impact_ow_vs_reduced_form.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/participation_rate_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/pair_1/figures/position_over_ADV_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_forced_liq/figures/total_net_pnl_by_pair.png

## 11. Caveats
- orderFlow_scenario = orderFlow_market + q_strategy.
- marginal impact bps = predicted_with_trade - predicted_market.
- Positive gross PnL means the strategy captures alpha before costs. Negative fitted net PnL can occur if model-implied impact costs exceed this gross alpha capture; that is not automatically a bug.
- Teammate coefficients are regression coefficients, not structural lambda/beta.
- reduced_form is regression, not dynamic AFS.
- x_flow is not silently treated as structural lambda.
- Costs are model-implied diagnostics.
- Reportable final results require full test months, not a max_rows debug sample.
