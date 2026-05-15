# Pair 2 Integration Report

- run_id: 20260515_015302
- figures_cleaned: True
- train_month: 201902
- test_month: 201903
- stocks: 20
- rows: 945501
- strategy_model: OW_transient_proxy
- total_net_pnl_reportable_strategy: 42379923.398321114
- max_participation_rate: 0.010000000000000002
- mean_participation_rate: 0.00019045917741368587
- max_abs_position: 707144.7028507807

Sizing controls:
- Previous unconstrained strategy variants can produce unrealistic trade sizes.
- The reported integrated strategy uses participation, trade, and inventory caps before wealth is interpreted.
- Fitted-model cost diagnostics should be read only after these caps are applied.

Teammate reduced_form is treated as a fitted regression evaluator, not dynamic AFS.
Wrong-model stress uses orderFlow_scenario = orderFlow_market + q_strategy.

No fitted proxy trades saved.
