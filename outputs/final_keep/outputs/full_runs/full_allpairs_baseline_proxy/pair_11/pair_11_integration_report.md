# Pair 11 Integration Report

- run_id: 20260515_012000
- figures_cleaned: True
- train_month: 201911
- test_month: 201912
- stocks: 20
- rows: 796918
- strategy_model: OW_transient_proxy
- total_net_pnl_reportable_strategy: 35510315.89270098
- max_participation_rate: 0.01
- mean_participation_rate: 0.0003504721269200852
- max_abs_position: 724488.5332584475

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
- total_gross_pnl_fitted_proxy: 36009191.564545065
- total_fitted_proxy_cost: 498875.67184408766
- total_net_pnl_fitted_proxy: 35510315.89270097
- total_turnover_fitted_proxy: 328152323.9157822
- max_participation_rate_fitted_proxy: 0.01
