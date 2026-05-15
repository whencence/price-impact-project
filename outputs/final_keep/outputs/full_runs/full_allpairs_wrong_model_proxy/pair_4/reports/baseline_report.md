# Pair 4 Integration Report

- run_id: 20260515_015302
- figures_cleaned: True
- train_month: 201904
- test_month: 201905
- stocks: 20
- rows: 984241
- strategy_model: OW_transient_proxy
- total_net_pnl_reportable_strategy: 54365250.30637565
- max_participation_rate: 0.01
- mean_participation_rate: 0.00023099427707668125
- max_abs_position: 874009.6461900576

Sizing controls:
- Previous unconstrained strategy variants can produce unrealistic trade sizes.
- The reported integrated strategy uses participation, trade, and inventory caps before wealth is interpreted.
- Fitted-model cost diagnostics should be read only after these caps are applied.

Teammate reduced_form is treated as a fitted regression evaluator, not dynamic AFS.
Wrong-model stress uses orderFlow_scenario = orderFlow_market + q_strategy.

No fitted proxy trades saved.
