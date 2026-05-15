# Teammate 2.1 / 2.2 / 2.3 Audit Report

## 1. Executive Summary
- Discovered 119 processed files across 2.1 and 2.2 folders.
- Processed 2.1 appears to contain monthly standardised bin/fill parquet files plus baseline 20-stock and rolling-universe artifacts.
- Processed 2.2 appears to contain rolling-pair fitted parameter files, metrics, logs, and model comparison tables.
- Main integration blocker: exact parameter units and reduced-form AFS convention must be confirmed before replacing my placeholders.

## 2. Processed Data Inventory
- Files by suffix: {'.csv': 85, '.parquet': 28, '.json': 3, '.png': 3}
- Largest files: data/processed_2_1/monthly_standardised/bin/bin_201905_standardised.parquet, data/processed_2_1/monthly_standardised/bin/bin_201908_standardised.parquet, data/processed_2_1/monthly_standardised/bin/bin_201910_standardised.parquet, data/processed_2_1/monthly_standardised/bin/bin_201903_standardised.parquet, data/processed_2_1/monthly_standardised/bin/bin_201907_standardised.parquet

## 3. Data Preparation Audit - Section 2.1
- Likely universe evidence: {'source_file': 'data/processed_2_1/baseline_20stocks/baseline_20stocks_summary.csv', 'stock_count': 20, 'stocks': 'AAL,AAP,AAPL,ABBV,ABT,ACN,ADBE,ADI,ADP,ADSK,AGN,ALGN,ALXN,AMAT,AMD,AMGN,AMT,AMZN,ANTM,APD', 'exactly_20': True, 'date_coverage': ''}
- Likely split/window evidence: {'source_file': 'data/processed_2_1/rolling_full_universe/rolling_pair_summary.csv', 'columns': 'pair_id|train_month|test_month|train_bin_path|test_bin_path|train_fill_path|test_fill_path|n_train_stocks|n_test_stocks|n_common_stocks_pair_universe|top20_common_stocks_by_train_notional', 'n_rows': 11, 'train_start': 201901, 'test_start': 201902, 'stock_count': nan}
- Missing values, duplicates, time gaps, and price jump summaries are saved as CSV outputs.

## 4. Model Fitting Audit - Section 2.2
- OW audit: [{'model_name': 'OW', 'implemented': 'yes', 'evidence_files': 'data/processed_2_2_rolling_baseline/parameters/ow_transient_params_by_pair_stock.csv', 'evidence_notebook_cells': 'notebooks/2_1_full_as_spec_completed.ipynb:cell0; notebooks/2_1_full_as_spec_completed.ipynb:cell2; notebooks/2_1_full_as_spec_completed.ipynb:cell7; notebooks/2_1_full_as_spec_completed.ipynb:cell10; notebooks/2_1_full_as_spec_completed.ipynb:cell11; notebooks/2_1_full_as_spec_completed.ipynb:cell12; notebooks/2_1_full_as_spec_completed.ipynb:cell13; notebooks/2_1_full_as_spec_completed.ipynb:cell15; notebooks/2_1_full_as_spec_completed.ipynb:cell17; notebooks/2_1_full_as_spec_completed.ipynb:cell18; notebooks/2_1_full_as_spec_completed.ipynb:cell19; notebooks/2_1_full_as_spec_completed.ipynb:cell23; notebooks/2_1_full_as_spec_completed.ipynb:cell24; notebooks/2_1_full_as_spec_completed.ipynb:cell28; notebooks/2_1_full_as_spec_completed.ipynb:cell31; notebooks/2_1_full_as_spec_completed.ipynb:cell34; notebooks/2_1_full_as_spec_completed.ipynb:cell35; notebooks/2_2_rolling_baseline_no_enhancement.ipynb:cell0; notebooks/2_2_rolling_baseline_no_enhancement.ipynb:cell1; notebooks/2_2_rolling_baseline_no_enhancement.ipynb:cell3', 'parameter_columns_found': 'pair_id|train_month|test_month|model|stock|half_life_sec|intercept|x_flow|ow_state_pre|x_trade|x_hidden|x_flow_depth|lobImb|effLobImb|spread_bps|source_file', 'parameter_scope': 'per_stock/per_window likely', 'normalization_used': 'unclear', 'beta_handling': 'parameter columns found', 'lambda_handling': 'unclear', 'notes': 'Inferred from filenames/columns/notebook keywords', 'concerns': ''}]
- Reduced-form / AFS audit: [{'model_name': 'reduced_form_AFS', 'implemented': 'yes', 'evidence_files': 'data/processed_2_2_rolling_baseline/parameters/reduced_form_params_by_pair_stock.csv', 'evidence_notebook_cells': 'notebooks/2_1_full_as_spec_completed.ipynb:cell0; notebooks/2_1_full_as_spec_completed.ipynb:cell2; notebooks/2_1_full_as_spec_completed.ipynb:cell7; notebooks/2_1_full_as_spec_completed.ipynb:cell10; notebooks/2_1_full_as_spec_completed.ipynb:cell11; notebooks/2_1_full_as_spec_completed.ipynb:cell12; notebooks/2_1_full_as_spec_completed.ipynb:cell13; notebooks/2_1_full_as_spec_completed.ipynb:cell15; notebooks/2_1_full_as_spec_completed.ipynb:cell17; notebooks/2_1_full_as_spec_completed.ipynb:cell18; notebooks/2_1_full_as_spec_completed.ipynb:cell19; notebooks/2_1_full_as_spec_completed.ipynb:cell23; notebooks/2_1_full_as_spec_completed.ipynb:cell24; notebooks/2_1_full_as_spec_completed.ipynb:cell28; notebooks/2_1_full_as_spec_completed.ipynb:cell31; notebooks/2_1_full_as_spec_completed.ipynb:cell34; notebooks/2_1_full_as_spec_completed.ipynb:cell35; notebooks/2_2_rolling_baseline_no_enhancement.ipynb:cell0; notebooks/2_2_rolling_baseline_no_enhancement.ipynb:cell1; notebooks/2_2_rolling_baseline_no_enhancement.ipynb:cell3', 'parameter_columns_found': 'pair_id|train_month|test_month|model|stock|half_life_sec|intercept|x_flow|ow_state_pre|x_trade|x_hidden|x_flow_depth|lobImb|effLobImb|spread_bps|source_file', 'parameter_scope': 'per_stock/per_window likely', 'normalization_used': 'unclear', 'beta_handling': 'parameter columns found', 'lambda_handling': 'unclear', 'notes': 'Inferred from filenames/columns/notebook keywords', 'concerns': ''}]
- Parameter files are listed in fitted_parameter_files.csv.

## 5. Backtest Engine Audit - Section 2.3
```text
                     item  status                                              evidence                               notes compatibility_with_my_code
           engine_present    PASS                               Notebook keyword search   Need inspect functions/signatures                    partial
             daily_resets UNCLEAR                 Search for reset/day/groupby evidence          Manual confirmation needed                    unknown
           generic_trades UNCLEAR          Search did not prove generic trade interface Ask teammate for function signature                    unknown
     waelbroeck_simulator UNCLEAR                                      Keyword evidence  May be absent or named differently                    partial
compatible_with_my_trades PARTIAL My trades include stock/date/timestamp/trade/position          Need exact required schema                    partial
```

## 6. Consistency with Course Specification
- Baseline 20-stock artifacts are present, and rolling pair artifacts are present.
- Whether the exact in-sample/out-of-sample split and same-stock universe are fully compliant should be confirmed from the config JSON/notebook cells.
- OW parameter files exist; normalization and beta/lambda units need final confirmation.
- Reduced-form parameter files exist; exact AFS reduction and v_t convention need final confirmation.

## 7. Compatibility with My Code
```text
                my_module                                          required_input found_in_teammate_outputs compatible                                                                     source_file                                           required_conversion                                                                          concerns                                              action_needed
      2.4 synthetic alpha                                     date,time,stock,mid                       yes        yes                                  processed_2_1 monthly_standardised bin parquet                    read parquet, preserve date/time/stock/mid                              Use full rolling window periods, not early rows only Point alpha generator at selected train/test parquet files
          2.5 OW strategy            alpha rows plus sigma/ADV and OW lambda/beta                       yes    partial processed_2_2_rolling_baseline/parameters/ow_transient_params_by_pair_stock.csv map fitted lambda/beta into OWStrategyConfig per stock/window My current config is global placeholder; needs loader for per-stock/window params             Confirm units of beta/lambda and normalization
2.5 reduced-form strategy lambda_base/beta or reduced-form params, v_t convention                       yes    partial processed_2_2_rolling_baseline/parameters/reduced_form_params_by_pair_stock.csv             write parameter loader and confirm v_t definition                            Reduced-form AFS exact formula/units need confirmation                 Ask teammate which reduced form was fitted
         2.7 stress tests         scenario-ready alpha/trades plus fitted configs                   partial    partial                                                  processed folders + my outputs                         rerun on chosen out-of-sample windows                           Stress results not final until calibrated params loaded                         Integrate fitted parameter loaders
```

## 8. Questions for Teammate
- Which exact dates are in-sample and out-of-sample for each rolling pair?
- Is the 20-stock universe fixed across windows or selected per pair?
- Which model is the reduced form of AFS exactly?
- Are lambda and beta global, per-stock, or rolling-window specific?
- What units are lambda and beta in, and is beta per minute?
- Is q raw signed volume or normalized volume in the fitted model?
- How is v_t computed?
- Which parameter file should I use for final strategy runs?
- Does the 2.3 backtest engine accept generic trade tables from my strategy modules?

## 9. Final Verdict
- Section 2.1: PASS
- Section 2.2: PARTIAL
- Section 2.3: PARTIAL
- Integration readiness: PARTIAL

## Integration TODO
```text
priority                                                                 task    owner                                        reason                 required_file_or_info
    high Confirm exact train/test rolling windows and fixed 20-stock universe teammate         Needed for final experiment alignment    rolling_pair_summary / config JSON
    high                       Confirm OW lambda/beta units and normalization teammate Needed to replace placeholder strategy params ow_transient_params_by_pair_stock.csv
    high                  Confirm reduced-form AFS formula and v_t definition teammate            Needed for enhanced model strategy reduced_form_params_by_pair_stock.csv
  medium         Write parameter loader from teammate outputs into my configs       me                              Integration step                        parameter CSVs
  medium                                 Confirm backtest engine trade schema     both    Needed to pass my strategy trades into 2.3                2_3 notebook/functions
```
