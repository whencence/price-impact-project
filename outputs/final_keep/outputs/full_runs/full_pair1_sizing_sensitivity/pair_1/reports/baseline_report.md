# Pair 1 Integration Report

- run_id: 20260514_191338
- figures_cleaned: True
- train_month: 201901
- test_month: 201902
- stocks: 20
- rows: 846332
- total_net_pnl_my_ow: 22170316.151852123
- max_participation_rate: 0.01
- mean_participation_rate: 0.0066169157297863886
- max_abs_position: 462046.46190476196

Sizing controls:
- Previous unconstrained strategy variants can produce unrealistic trade sizes.
- The reported integrated strategy uses participation, trade, and inventory caps before wealth is interpreted.
- Fitted-model cost diagnostics should be read only after these caps are applied.

Teammate reduced_form is treated as a fitted regression evaluator, not dynamic AFS.
Wrong-model stress uses orderFlow_scenario = orderFlow_market + q_strategy.

No fitted proxy trades saved.
