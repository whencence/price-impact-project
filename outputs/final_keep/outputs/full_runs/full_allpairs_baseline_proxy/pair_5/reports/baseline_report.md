# Pair 5 Integration Report

- run_id: 20260515_012000
- figures_cleaned: True
- train_month: 201905
- test_month: 201906
- stocks: 20
- rows: 866098
- strategy_model: OW_transient_proxy
- total_net_pnl_reportable_strategy: 60123358.96210504
- max_participation_rate: 0.01
- mean_participation_rate: 0.0002005321018063436
- max_abs_position: 864963.09422619

Sizing controls:
- Previous unconstrained strategy variants can produce unrealistic trade sizes.
- The reported integrated strategy uses participation, trade, and inventory caps before wealth is interpreted.
- Fitted-model cost diagnostics should be read only after these caps are applied.

Teammate reduced_form is treated as a fitted regression evaluator, not dynamic AFS.
Wrong-model stress uses orderFlow_scenario = orderFlow_market + q_strategy.

Fitted-regression proxy strategy:
- Because teammate's fitted impact models are regressions rather than structural control models, proxy strategies are built from local marginal impact slopes.
- Current reportable strategy: OW_transient_proxy.
- This is not a closed-form dynamic optimal strategy under a structural impact model.
- total_gross_pnl_fitted_proxy: 60793223.76845351
- total_fitted_proxy_cost: 669864.8063484582
- total_net_pnl_fitted_proxy: 60123358.96210504
- total_turnover_fitted_proxy: 319581783.4299774
- max_participation_rate_fitted_proxy: 0.01
