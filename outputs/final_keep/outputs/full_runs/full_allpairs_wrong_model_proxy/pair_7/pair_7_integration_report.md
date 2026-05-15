# Pair 7 Integration Report

- run_id: 20260515_015302
- figures_cleaned: True
- train_month: 201907
- test_month: 201908
- stocks: 20
- rows: 958551
- strategy_model: OW_transient_proxy
- total_net_pnl_reportable_strategy: 74247055.6272873
- max_participation_rate: 0.01
- mean_participation_rate: 0.00024065861558156035
- max_abs_position: 701686.5637723427

Sizing controls:
- Previous unconstrained strategy variants can produce unrealistic trade sizes.
- The reported integrated strategy uses participation, trade, and inventory caps before wealth is interpreted.
- Fitted-model cost diagnostics should be read only after these caps are applied.

Teammate reduced_form is treated as a fitted regression evaluator, not dynamic AFS.
Wrong-model stress uses orderFlow_scenario = orderFlow_market + q_strategy.

No fitted proxy trades saved.
