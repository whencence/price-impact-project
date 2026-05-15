# Integrated Rolling Simulation Report

## 1. Executive Summary
- Run ID: 20260515_070909
- Figures cleaned before run: True
- Run-specific archive: /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/runs/20260515_070909
- Pair IDs: [1]
- Truncated debug run: False
- Row-level trade files saved: False
- Reportable strategy model: OW_transient_proxy
- Current reportable strategy: OW_transient_proxy.
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
 pair_id     strategy_model  n_rows  n_stocks  number_of_unique_dates  total_gross_pnl  total_net_pnl  total_signed_impact_cost_normalized  total_quadratic_impact_cost_normalized  mean_daily_net_pnl  std_daily_net_pnl  daily_sharpe  annualized_sharpe  total_signed_volume_turnover  total_notional_turnover  average_daily_turnover_shares  average_daily_turnover_notional  total_normalized_turnover  max_participation_rate  mean_participation_rate  max_drawdown  max_daily_drawdown  max_abs_position  max_abs_impact  number_of_trades  number_of_stock_days  mean_abs_trade  median_abs_trade   rows  rows_with_scaling  rows_skipped_due_to_missing_scaling
       1 OW_transient_proxy  846332        20                      19     3.302339e+07   3.252236e+07                                  0.0                           501020.525605        1.711703e+06      909909.450817       1.88118          29.862806                  2.826172e+08             4.278691e+10                   1.487459e+07                     2.251943e+09                 135.925557                    0.01                 0.000161   -1670631.88                 0.0     705447.302705        0.139737            793919                   380      333.931868        149.573831 846332             846332                                    0
- Daily Sharpe is meaningful for this run.

## 3. Alpha Diagnostics
- Baseline synthetic alpha: h=5m, rho=0.10, H_alpha=5m unless configured otherwise.
- alpha_mean_bps: 0.0010
- alpha_std_bps: 0.2322
- corr(alpha, future_return_h): 0.1904
- corr(position_before, future_return_h): 0.0277
- corr(trade, alpha): 0.4899
- corr(position_after, alpha): 0.5924
- share_sign_position_matches_alpha: 0.6528
- share_sign_trade_matches_alpha: 0.8575
- gross alpha capture: 3.3023e+07
- Interpretation: The strategy captures alpha before fitted impact costs.

## 4. Strategy Sizing Diagnostics
- mean_abs_trade: 333.9319
- median_abs_trade: 149.5738
- max_abs_trade: 95503.4676
- max_abs_position: 7.0545e+05
- total_signed_volume_turnover: 2.8262e+08
- total_notional_turnover: 4.2787e+10
- max_participation_rate: 0.0100
- mean_participation_rate: 1.6061e-04
- share_trade_clipped: 0.0639
- share_position_clipped: 0.0634
- max_position_over_ADV: 0.0500
- Previous unconstrained runs were unrealistic. The capped runs are the reportable ones.

## 5. Baseline Fitted OW_transient Proxy Strategy
- total_gross_pnl: 3.3023e+07
- total_local_proxy_cost: 5.0102e+05
- total_net_pnl_local_proxy: 3.2522e+07
- local_cost_to_gross_pnl_ratio: 0.0152
- max_drawdown: -1.6706e+06
- Reportable strategy: OW_transient_proxy.
- Legacy theoretical OW was not run.

## 6. Fitted Regression Evaluator Wealth
- total_fitted_cost_ow_regression: 6.2476e+05
- total_abs_fitted_cost_ow_regression: 1.2226e+06
- net_pnl_under_ow_regression_eval: 3.2399e+07
- mean_abs_marginal_impact_ow_regression_bps: 0.2833
- total_fitted_cost_reduced_form: 6.7381e+05
- total_abs_fitted_cost_reduced_form: 2.4571e+06
- net_pnl_under_reduced_form_eval: 3.2350e+07
- mean_abs_marginal_impact_reduced_form_bps: 0.2988
- Interpretation: Fitted impact costs do not overturn gross PnL on this sample.
- Model comparison: Reduced-form evaluator is more punitive than OW_transient on this sample.

## 8. PnL Formula and Timing Validation
                                                check status                                                                                                                                                                                                                                                                          message  pair_id
                         reportable_strategy_is_proxy   PASS                                                                                                                                                                                                                                                strategy_model=OW_transient_proxy      1.0
                     baseline_uses_OW_transient_proxy   PASS                                                                                                                                                                                                                                     reportable strategy_model=OW_transient_proxy      1.0
                                delta_mid_consistency   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                       gross_pnl_uses_position_before   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                                    position_dynamics   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                                 daily_position_reset   PASS                                                                                                                                                                                                                                                         max_first_abs_position=0      1.0
                                       final_position   WARN                                                                                                                                                                                                                                     max_final_abs_position=5.47e+04; capped=True      1.0
                                no_lookahead_pnl_note   PASS                                                                                                                                                                               gross PnL is position_before*delta_mid; aggregate difference vs position_after convention=8.73e+06      1.0
                              alpha_trade_timing_note   PASS                                                                                                                                                                               trade at row t uses alpha_t and affects position_after_t; PnL over t-1 to t uses position_before_t      1.0
                             alpha_future_return_corr   PASS                                                                                                                                                                                                                                                                      corr=0.1904      1.0
                          position_future_return_corr   PASS                                                                                                                                                                                                                                                                     corr=0.02769      1.0
                                     trade_alpha_corr   PASS                                                                                                                                                                                                                              corr_trade_alpha=0.5366; corr_position_alpha=0.5929      1.0
                                       sign_alignment   PASS                                                                                                                                                                                                  share_position_same_sign_as_alpha=65.28%; share_trade_same_sign_as_alpha=85.75%      1.0
                                  gross_alpha_capture   PASS                                                                                                                                                                                                                                                            gross_pnl=3.30234e+07      1.0
         ow_regression_orderflow_market_plus_strategy   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                ow_regression_marginal_impact_formula   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                   ow_regression_bps_to_price_formula   PASS                                                                                                                                                                                                                                                               max_error=5.55e-17      1.0
                           ow_regression_cost_formula   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                       ow_regression_cost_sign_sanity   PASS                                                                                                                                                          share_same_direction=59.79%; share_positive_cost=59.79%; total_signed=624763; total_abs=1.22263e+06; netout_ratio=0.511      1.0
            ow_regression_cost_subtraction_convention   PASS                                                                                                                                                                     net_pnl_fitted_model = gross_pnl - fitted_impact_cost_signed; positive fitted cost is adverse and subtracted      1.0
          reduced_form_orderflow_market_plus_strategy   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                 reduced_form_marginal_impact_formula   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                    reduced_form_bps_to_price_formula   PASS                                                                                                                                                                                                                                                               max_error=4.44e-16      1.0
                            reduced_form_cost_formula   PASS                                                                                                                                                                                                                                                                      max_error=0      1.0
                        reduced_form_cost_sign_sanity   PASS                                                                                                                                                          share_same_direction=57.79%; share_positive_cost=57.79%; total_signed=673812; total_abs=2.45706e+06; netout_ratio=0.274      1.0
             reduced_form_cost_subtraction_convention   PASS                                                                                                                                                                     net_pnl_fitted_model = gross_pnl - fitted_impact_cost_signed; positive fitted cost is adverse and subtracted      1.0
                                    participation_cap   PASS                                                                                                                                                                                                                                             max=0.01; mean=0.000160605; cap=0.01      1.0
                                    position_over_adv   PASS                                                                                                                                                                                                                                                                         max=0.05      1.0
                                  trade_clipped_share   PASS                                                                                                                                                                                                                                                                      share=6.39%      1.0
                         fitted_marginal_impact_scale   PASS                                                                                                                                                                                                                 mean_abs=0.2988; median_abs=0.1502; p95_abs=1.076; max_abs=219.2      1.0
                                    notional_turnover   PASS                                                                                                                                                                                                                                              total_notional_turnover=4.27869e+10      1.0
                       sizing_uses_OW_transient_proxy   PASS                                                                                                                                                                                                                                                strategy_model=OW_transient_proxy      1.0
                        no_legacy_ow_in_final_outputs   PASS                                                                                                                                                                                                                                                     legacy_outputs_present=False      1.0
                       all_pairs_strategy_summary.csv   PASS                                                                                             /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/all_pairs_strategy_summary.csv      NaN
                    all_pairs_sensitivity_summary.csv   PASS                                                                                          /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/all_pairs_sensitivity_summary.csv      NaN
                         all_selected_pairs_processed   PASS                                                                                                                                                                                                                                                                       missing=[]      NaN
                                        wealth_finite   PASS                                                                                                                                                                                                                                                             aggregate pnl finite      NaN
                          participation_rate_reported   PASS                                                                                                                                                                                                                                                      max_participation_rate=0.01      NaN
                     total_notional_turnover_reported   PASS                                                                                                                                                                                                                                              total_notional_turnover=4.27869e+10      NaN
                     figure_total_net_pnl_by_pair.png   PASS                             path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/figures/total_net_pnl_by_pair.png; exists=True; size=20175; modified_after_run_start=True      NaN
figure_pair_cumulative_wealth_reportable_strategy.png   PASS path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/pair_cumulative_wealth_reportable_strategy.png; exists=True; size=45377; modified_after_run_start=True      NaN
              figure_pair_fitted_evaluator_wealth.png   PASS               path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/pair_fitted_evaluator_wealth.png; exists=True; size=51597; modified_after_run_start=True      NaN
      figure_cumulative_wealth_internal_vs_fitted.png   PASS       path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/cumulative_wealth_internal_vs_fitted.png; exists=True; size=58537; modified_after_run_start=True      NaN
       figure_gross_pnl_vs_fitted_cost_cumulative.png   PASS        path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/gross_pnl_vs_fitted_cost_cumulative.png; exists=True; size=57329; modified_after_run_start=True      NaN
             figure_marginal_impact_bps_histogram.png   PASS              path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/marginal_impact_bps_histogram.png; exists=True; size=29938; modified_after_run_start=True      NaN
              figure_participation_rate_histogram.png   PASS               path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/participation_rate_histogram.png; exists=True; size=34859; modified_after_run_start=True      NaN
               figure_position_over_ADV_histogram.png   PASS                path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/position_over_ADV_histogram.png; exists=True; size=32355; modified_after_run_start=True      NaN
                 portfolio_internal_unique_timestamps   SKIP                                                                                                                                                                                                           row-level internal trades not saved; summary metrics validated instead      1.0
       portfolio_internal_final_matches_total_net_pnl   SKIP                                                                                                                                                                                                                                              row-level internal trades not saved      1.0
                            portfolio_plot_data_build   SKIP                                                                                                                                                                                                                     row-level trade/evaluator files not saved for reconciliation      1.0

## 12. Sizing Sensitivity
 pair_id     strategy_model  target_impact_scale  proxy_trade_scale  max_participation_rate  total_net_pnl  total_turnover  max_participation_rate_realized  mean_participation_rate  max_abs_position  share_trade_clipped  share_position_clipped  total_fitted_cost_ow_regression  net_pnl_under_ow_regression_eval  total_fitted_cost_reduced_form  net_pnl_under_reduced_form_eval
       1 OW_transient_proxy                 0.10               0.10                   0.005   3.665808e+06    3.090299e+07                            0.005                 0.000020      70544.730270             0.000497                0.000048                     27441.976139                      3.652427e+06                    52618.489794                     3.627251e+06
       1 OW_transient_proxy                 0.10               0.10                   0.010   3.660538e+06    3.096112e+07                            0.010                 0.000021      70544.730270             0.000497                0.000048                     35090.037404                      3.644779e+06                    69088.917895                     3.610780e+06
       1 OW_transient_proxy                 0.10               0.10                   0.020   3.660538e+06    3.096112e+07                            0.010                 0.000021      70544.730270             0.000497                0.000048                     35090.037404                      3.644779e+06                    69088.917895                     3.610780e+06
       1 OW_transient_proxy                 0.25               0.25                   0.005   9.793634e+06    7.627436e+07                            0.005                 0.000049     176361.825676             0.003103                0.002654                     83247.987663                      9.762629e+06                   119280.082966                     9.726597e+06
       1 OW_transient_proxy                 0.25               0.25                   0.010   9.769514e+06    7.659319e+07                            0.010                 0.000050     176361.825676             0.003103                0.002654                    113533.433706                      9.732344e+06                   166611.514922                     9.679266e+06
       1 OW_transient_proxy                 0.25               0.25                   0.020   9.769514e+06    7.659319e+07                            0.010                 0.000050     176361.825676             0.003103                0.002654                    113533.433706                      9.732344e+06                   166611.514922                     9.679266e+06
       1 OW_transient_proxy                 0.50               0.50                   0.005   1.878353e+07    1.482389e+08                            0.005                 0.000092     352723.651352             0.018121                0.017616                    202990.215422                      1.872188e+07                   242977.980971                     1.868189e+07
       1 OW_transient_proxy                 0.50               0.50                   0.010   1.877053e+07    1.489378e+08                            0.010                 0.000093     352723.651352             0.018076                0.017630                    258703.635384                      1.870530e+07                   315190.945010                     1.864882e+07
       1 OW_transient_proxy                 0.50               0.50                   0.020   1.877053e+07    1.489378e+08                            0.010                 0.000093     352723.651352             0.018076                0.017630                    258703.635384                      1.870530e+07                   315190.945010                     1.864882e+07
       1 OW_transient_proxy                 1.00               1.00                   0.005   3.250656e+07    2.812351e+08                            0.005                 0.000159     705447.302705             0.063994                0.063383                    518526.231184                      3.238204e+07                   562506.762521                     3.233806e+07
       1 OW_transient_proxy                 1.00               1.00                   0.010   3.252236e+07    2.826172e+08                            0.010                 0.000161     705447.302705             0.063903                0.063446                    624763.372498                      3.239862e+07                   673811.662720                     3.234957e+07
       1 OW_transient_proxy                 1.00               1.00                   0.020   3.252236e+07    2.826172e+08                            0.010                 0.000161     705447.302705             0.063903                0.063446                    624763.372498                      3.239862e+07                   673811.662720                     3.234957e+07

## 13. Figures
- Portfolio-level plots aggregate PnL and costs across all stocks at each timestamp before taking cumulative sums.
- Sample path plots explicitly use one selected stock/day only.
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/alpha_position_alignment_sample.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/clipped_trade_share_by_day.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/cumulative_wealth_internal_vs_fitted.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/drawdown_internal_vs_fitted.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/fitted_cost_vs_gross_pnl_scatter.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/gross_pnl_vs_fitted_cost_cumulative.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/marginal_impact_bps_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/marginal_impact_bps_timeseries_sample.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/pair_cumulative_wealth_reportable_strategy.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/pair_fitted_evaluator_wealth.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/pair_marginal_impact_ow_vs_reduced_form.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/participation_rate_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/position_over_ADV_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/sizing_sensitivity_net_pnl.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/pair_1/figures/sizing_sensitivity_turnover.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/final_pair1_sizing_sensitivity_proxy/figures/total_net_pnl_by_pair.png

## 14. Caveats
- orderFlow_scenario = orderFlow_market + q_strategy.
- marginal impact bps = predicted_with_trade - predicted_market.
- Positive gross PnL means the strategy captures alpha before costs. Negative fitted net PnL can occur if model-implied impact costs exceed this gross alpha capture; that is not automatically a bug.
- Teammate coefficients are regression coefficients, not structural lambda/beta.
- reduced_form is regression, not dynamic AFS.
- x_flow is not silently treated as structural lambda.
- Costs are model-implied diagnostics.
- Reportable final results require full test months, not a max_rows debug sample.
