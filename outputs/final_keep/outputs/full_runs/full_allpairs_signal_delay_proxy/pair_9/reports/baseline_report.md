# Pair 9 Integration Report

- run_id: 20260515_032727
- figures_cleaned: True
- train_month: 201909
- test_month: 201910
- stocks: 20
- rows: 1022167
- strategy_model: OW_transient_proxy
- total_net_pnl_reportable_strategy: 105004628.59105286
- max_participation_rate: 0.01
- mean_participation_rate: 0.00027073975014474383
- max_abs_position: 933914.0055120457

Sizing controls:
- Previous unconstrained strategy variants can produce unrealistic trade sizes.
- The reported integrated strategy uses participation, trade, and inventory caps before wealth is interpreted.
- Fitted-model cost diagnostics should be read only after these caps are applied.

Teammate reduced_form is treated as a fitted regression evaluator, not dynamic AFS.
Wrong-model stress uses orderFlow_scenario = orderFlow_market + q_strategy.

No fitted proxy trades saved.
