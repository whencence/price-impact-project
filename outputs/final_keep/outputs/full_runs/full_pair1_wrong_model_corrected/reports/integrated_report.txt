# Integrated Rolling Simulation Report

## 1. Executive Summary
- Run ID: 20260515_001351
- Figures cleaned before run: True
- Run-specific archive: /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/runs/20260515_001351
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
- alpha_mean_bps: n/a
- alpha_std_bps: n/a
- corr(alpha, future_return_h): n/a
- corr(position_before, future_return_h): n/a
- corr(trade, alpha): n/a
- corr(position_after, alpha): n/a
- share_sign_position_matches_alpha: n/a
- share_sign_trade_matches_alpha: n/a
- gross alpha capture: n/a
- Interpretation: The strategy captures alpha before fitted impact costs.

## 4. Strategy Sizing Diagnostics
- mean_abs_trade: 7618.1773
- median_abs_trade: 2130.7784
- max_abs_trade: n/a
- max_abs_position: 4.6205e+05
- total_signed_volume_turnover: 6.4475e+09
- total_notional_turnover: 4.5611e+11
- max_participation_rate: 0.0100
- mean_participation_rate: 0.0066
- share_trade_clipped: nan
- share_position_clipped: nan
- max_position_over_ADV: nan
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
- Interpretation: The alpha strategy is profitable before fitted impact costs, but fitted regression impact costs more than offset gross PnL on this sample.
- Model comparison: Reduced-form evaluator is less punitive than OW_transient on this sample.

## 8. PnL Formula and Timing Validation
                                                        check status                                                                                                                                                                                                                                                                        message  pair_id
                                        delta_mid_consistency   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                               gross_pnl_uses_position_before   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                                            position_dynamics   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                                         daily_position_reset   PASS                                                                                                                                                                                                                                                       max_first_abs_position=0      1.0
                                               final_position   WARN                                                                                                                                                                                                                                    max_final_abs_position=3.7e+05; capped=True      1.0
                                        no_lookahead_pnl_note   PASS                                                                                                                                                                             gross PnL is position_before*delta_mid; aggregate difference vs position_after convention=7.36e+07      1.0
                                      alpha_trade_timing_note   PASS                                                                                                                                                                             trade at row t uses alpha_t and affects position_after_t; PnL over t-1 to t uses position_before_t      1.0
                                     alpha_future_return_corr   PASS                                                                                                                                                                                                                                                                    corr=0.1904      1.0
                                  position_future_return_corr   WARN                                                                                                                                                                                                                                                                  corr=-0.03486      1.0
                                             trade_alpha_corr   PASS                                                                                                                                                                                                                         corr_trade_alpha=0.005467; corr_position_alpha=0.03807      1.0
                                               sign_alignment   PASS                                                                                                                                                                                                share_position_same_sign_as_alpha=59.22%; share_trade_same_sign_as_alpha=53.06%      1.0
                                          gross_alpha_capture   PASS                                                                                                                                                                                                                                                          gross_pnl=2.21703e+07      1.0
                 ow_regression_orderflow_market_plus_strategy   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                        ow_regression_marginal_impact_formula   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                           ow_regression_bps_to_price_formula   PASS                                                                                                                                                                                                                                                             max_error=5.55e-17      1.0
                                   ow_regression_cost_formula   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                               ow_regression_cost_sign_sanity   PASS                                                                                                                                                   share_same_direction=92.16%; share_positive_cost=92.16%; total_signed=7.62243e+07; total_abs=7.72723e+07; netout_ratio=0.986      1.0
                    ow_regression_cost_subtraction_convention   PASS                                                                                                                                                                   net_pnl_fitted_model = gross_pnl - fitted_impact_cost_signed; positive fitted cost is adverse and subtracted      1.0
                  reduced_form_orderflow_market_plus_strategy   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                         reduced_form_marginal_impact_formula   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                            reduced_form_bps_to_price_formula   PASS                                                                                                                                                                                                                                                             max_error=4.44e-16      1.0
                                    reduced_form_cost_formula   PASS                                                                                                                                                                                                                                                                    max_error=0      1.0
                                reduced_form_cost_sign_sanity   PASS                                                                                                                                                   share_same_direction=90.58%; share_positive_cost=90.58%; total_signed=7.46947e+07; total_abs=7.70841e+07; netout_ratio=0.969      1.0
                     reduced_form_cost_subtraction_convention   PASS                                                                                                                                                                   net_pnl_fitted_model = gross_pnl - fitted_impact_cost_signed; positive fitted cost is adverse and subtracted      1.0
                                            participation_cap   PASS                                                                                                                                                                                                                                            max=0.01; mean=0.00661692; cap=0.01      1.0
                                            position_over_adv   PASS                                                                                                                                                                                                                                                                       max=0.05      1.0
                                          trade_clipped_share   WARN                                                                                                                                                                                                                                                                   share=82.50%      1.0
                                 fitted_marginal_impact_scale   PASS                                                                                                                                                                                                                  mean_abs=0.8907; median_abs=0.558; p95_abs=2.983; max_abs=261      1.0
                                            notional_turnover   PASS                                                                                                                                                                                                                                            total_notional_turnover=4.56111e+11      1.0
                             same_trade_evaluator_sensitivity   PASS                                                                                                                                                                                                                               same OW theoretical trades under both evaluators      1.0
                                proxy_strategy_model_distinct   PASS                                                                                                                                                                                                                 OW_transient_proxy and reduced_form_proxy trade paths compared      1.0
                                  wrong_model_matrix_complete   PASS                                                                                                                                                                                                                                                                        cells=4      1.0
                                     wrong_model_loss_formula   PASS                                                                                                                                                                                                                               loss = wrong_strategy_pnl - correct_strategy_pnl      1.0
           proxy_strategy_cost_consistency_OW_transient_proxy   PASS                                                                                                                                                                                                                               full_evaluator/internal_proxy_cost_ratio=1.24698      1.0
           proxy_strategy_cost_consistency_reduced_form_proxy   PASS                                                                                                                                                                                                                              full_evaluator/internal_proxy_cost_ratio=0.984118      1.0
orderFlow_scenario_convention_OW_transient_proxy_OW_transient   PASS                                                                                                                                                                                                                             orderFlow_scenario = orderFlow_market + q_strategy      1.0
orderFlow_scenario_convention_OW_transient_proxy_reduced_form   PASS                                                                                                                                                                                                                             orderFlow_scenario = orderFlow_market + q_strategy      1.0
orderFlow_scenario_convention_reduced_form_proxy_OW_transient   PASS                                                                                                                                                                                                                             orderFlow_scenario = orderFlow_market + q_strategy      1.0
orderFlow_scenario_convention_reduced_form_proxy_reduced_form   PASS                                                                                                                                                                                                                             orderFlow_scenario = orderFlow_market + q_strategy      1.0
                               all_pairs_strategy_summary.csv   PASS                                                                                               /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/all_pairs_strategy_summary.csv      NaN
                            all_pairs_wrong_model_summary.csv   PASS                                                                                            /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/all_pairs_wrong_model_summary.csv      NaN
                                 all_selected_pairs_processed   PASS                                                                                                                                                                                                                                                                     missing=[]      NaN
                                                wealth_finite   PASS                                                                                                                                                                                                                                                           aggregate pnl finite      NaN
                                  participation_rate_reported   PASS                                                                                                                                                                                                                                                    max_participation_rate=0.01      NaN
                             total_notional_turnover_reported   PASS                                                                                                                                                                                                                                            total_notional_turnover=4.56111e+11      NaN
                             figure_total_net_pnl_by_pair.png   PASS                               path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/figures/total_net_pnl_by_pair.png; exists=True; size=16405; modified_after_run_start=True      NaN
                      figure_pair_cumulative_wealth_my_ow.png   PASS                 path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/pair_cumulative_wealth_my_ow.png; exists=True; size=38973; modified_after_run_start=True      NaN
                      figure_pair_fitted_evaluator_wealth.png   PASS                 path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/pair_fitted_evaluator_wealth.png; exists=True; size=46761; modified_after_run_start=True      NaN
              figure_cumulative_wealth_internal_vs_fitted.png   PASS         path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/cumulative_wealth_internal_vs_fitted.png; exists=True; size=54288; modified_after_run_start=True      NaN
               figure_gross_pnl_vs_fitted_cost_cumulative.png   PASS          path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/gross_pnl_vs_fitted_cost_cumulative.png; exists=True; size=56516; modified_after_run_start=True      NaN
                     figure_marginal_impact_bps_histogram.png   PASS                path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/marginal_impact_bps_histogram.png; exists=True; size=24811; modified_after_run_start=True      NaN
                      figure_participation_rate_histogram.png   PASS                 path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/participation_rate_histogram.png; exists=True; size=26677; modified_after_run_start=True      NaN
                       figure_position_over_ADV_histogram.png   PASS                  path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/position_over_ADV_histogram.png; exists=True; size=27053; modified_after_run_start=True      NaN
                        figure_wrong_model_net_pnl_matrix.png   PASS                   path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/wrong_model_net_pnl_matrix.png; exists=True; size=36342; modified_after_run_start=True      NaN
                           figure_wrong_model_cost_matrix.png   PASS                      path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/wrong_model_cost_matrix.png; exists=True; size=44350; modified_after_run_start=True      NaN
                  figure_wrong_model_turnover_by_strategy.png   PASS             path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/wrong_model_turnover_by_strategy.png; exists=True; size=26153; modified_after_run_start=True      NaN
              figure_wrong_model_cumulative_wealth_matrix.png   PASS         path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/wrong_model_cumulative_wealth_matrix.png; exists=True; size=68715; modified_after_run_start=True      NaN
      figure_impact_evaluator_sensitivity_same_trade_path.png   PASS path=/Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/impact_evaluator_sensitivity_same_trade_path.png; exists=True; size=26828; modified_after_run_start=True      NaN
                         portfolio_internal_unique_timestamps   SKIP                                                                                                                                                                                                         row-level internal trades not saved; summary metrics validated instead      1.0
               portfolio_internal_final_matches_total_net_pnl   SKIP                                                                                                                                                                                                                                            row-level internal trades not saved      1.0
                                    portfolio_plot_data_build   SKIP                                                                                                                                                                                                                   row-level trade/evaluator files not saved for reconciliation      1.0

## 9. Wrong Model Stress
### 9.1 Impact Evaluator Sensitivity
- Same OW course/theoretical trade path evaluated under OW_transient and reduced_form fitted regressions.
- This is evaluator sensitivity, not full wrong-model strategy optimization.
 pair_id                                   scenario_name                scenario_type  total_fitted_cost_ow_regression  total_abs_fitted_cost_ow_regression  net_pnl_under_ow_regression_eval  mean_abs_marginal_impact_ow_regression_bps  total_fitted_cost_reduced_form  total_abs_fitted_cost_reduced_form  net_pnl_under_reduced_form_eval  mean_abs_marginal_impact_reduced_form_bps  total_cost_difference_rf_minus_ow  net_pnl_difference_rf_minus_ow  degradation_rf_vs_ow  corr_between_ow_and_rf_marginal_impact
       1 impact_evaluator_sensitivity_same_ow_trade_path impact_evaluator_sensitivity                     7.622430e+07                         7.727228e+07                     -5.405398e+07                                    0.889748                    7.469467e+07                        7.708409e+07                    -5.252436e+07                                   0.890653                      -1.529625e+06                    1.529625e+06          1.529625e+06                                0.848817

### 9.2 Fitted Proxy Strategies By Assumed Model
- Fitted models are regressions, so strategy generation uses local myopic quadratic-cost proxies under each assumed model.
 pair_id     strategy_model assumed_model  total_turnover  total_notional_turnover  max_participation_rate  mean_participation_rate  total_internal_proxy_cost  total_internal_proxy_net_pnl
       1 OW_transient_proxy  OW_transient    2.826172e+08             4.278691e+10                    0.01                 0.000161              501020.525605                  3.252236e+07
       1 reduced_form_proxy  reduced_form    3.059006e+08             4.252267e+10                    0.01                 0.000187              465404.544339                  3.033579e+07

### 9.3 True Wrong-Model Strategy Misspecification Matrix
- Rows are strategy trade paths generated under the assumed fitted model.
- Columns are fitted evaluator models treated as the true realized impact model.
 pair_id     strategy_model assumed_model evaluator_model   true_model  total_gross_pnl  total_fitted_cost  total_abs_fitted_cost      net_pnl  total_turnover  total_notional_turnover  cost_bps_of_turnover  max_drawdown  daily_sharpe  mean_abs_marginal_impact_bps  max_participation_rate  mean_participation_rate
       1 OW_transient_proxy  OW_transient    OW_transient OW_transient     3.302339e+07      624763.372498           1.222635e+06 3.239862e+07    2.826172e+08             4.278691e+10              0.146017 -1.729035e+06      1.876628                      0.283250                    0.01                 0.000161
       1 OW_transient_proxy  OW_transient    reduced_form reduced_form     3.302339e+07      673811.662720           2.457055e+06 3.234957e+07    2.826172e+08             4.278691e+10              0.157481 -1.711652e+06      1.878248                      0.298765                    0.01                 0.000161
       1 reduced_form_proxy  reduced_form    OW_transient OW_transient     3.080120e+07      761398.892734           1.382817e+06 3.003980e+07    3.059006e+08             4.252267e+10              0.179057 -1.669562e+06      1.736425                      0.288338                    0.01                 0.000187
       1 reduced_form_proxy  reduced_form    reduced_form reduced_form     3.080120e+07      458012.829106           2.370838e+06 3.034319e+07    3.059006e+08             4.252267e+10              0.107710 -1.652595e+06      1.763341                      0.297826                    0.01                 0.000187

### 9.4 Wrong-Model Losses
 pair_id   true_model correct_strategy_model wrong_strategy_model  correct_model_pnl  wrong_model_pnl  wrong_model_loss
       1 reduced_form     reduced_form_proxy   OW_transient_proxy       3.034319e+07     3.234957e+07      2.006388e+06
       1 OW_transient     OW_transient_proxy   reduced_form_proxy       3.239862e+07     3.003980e+07     -2.358823e+06

- If true model is reduced_form, using OW_transient proxy instead of reduced_form proxy changes PnL by 2.0064e+06.
- If true model is OW_transient, using reduced_form proxy instead of OW_transient proxy changes PnL by -2.3588e+06.

## 13. Figures
- Portfolio-level plots aggregate PnL and costs across all stocks at each timestamp before taking cumulative sums.
- Sample path plots explicitly use one selected stock/day only.
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/alpha_position_alignment_sample.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/clipped_trade_share_by_day.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/cumulative_wealth_internal_vs_fitted.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/drawdown_internal_vs_fitted.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/fitted_cost_vs_gross_pnl_scatter.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/gross_pnl_vs_fitted_cost_cumulative.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/impact_evaluator_sensitivity_same_trade_path.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/marginal_impact_bps_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/marginal_impact_bps_timeseries_sample.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/pair_cumulative_wealth_my_ow.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/pair_fitted_evaluator_wealth.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/pair_marginal_impact_ow_vs_reduced_form.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/participation_rate_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/position_over_ADV_histogram.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/wrong_model_cost_matrix.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/wrong_model_cumulative_wealth_matrix.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/wrong_model_net_pnl_matrix.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/pair_1/figures/wrong_model_turnover_by_strategy.png
- /Users/ulysse/Desktop/Studies/Imperial/Computing for finance/Price Impact/price-impact-project/outputs/full_runs/full_pair1_wrong_model_corrected/figures/total_net_pnl_by_pair.png

## 14. Caveats
- orderFlow_scenario = orderFlow_market + q_strategy.
- marginal impact bps = predicted_with_trade - predicted_market.
- Positive gross PnL means the strategy captures alpha before costs. Negative fitted net PnL can occur if model-implied impact costs exceed this gross alpha capture; that is not automatically a bug.
- Teammate coefficients are regression coefficients, not structural lambda/beta.
- reduced_form is regression, not dynamic AFS.
- x_flow is not silently treated as structural lambda.
- Costs are model-implied diagnostics.
- Reportable final results require full test months, not a max_rows debug sample.
