# Pair 6 Integration Report

- run_id: 20260515_032727
- figures_cleaned: True
- train_month: 201906
- test_month: 201907
- stocks: 20
- rows: 944894
- strategy_model: OW_transient_proxy
- total_net_pnl_reportable_strategy: 38849660.41777764
- max_participation_rate: 0.01
- mean_participation_rate: 0.0002399154378115721
- max_abs_position: 908655.6863461448

Sizing controls:
- Previous unconstrained strategy variants can produce unrealistic trade sizes.
- The reported integrated strategy uses participation, trade, and inventory caps before wealth is interpreted.
- Fitted-model cost diagnostics should be read only after these caps are applied.

Teammate reduced_form is treated as a fitted regression evaluator, not dynamic AFS.
Wrong-model stress uses orderFlow_scenario = orderFlow_market + q_strategy.

No fitted proxy trades saved.
