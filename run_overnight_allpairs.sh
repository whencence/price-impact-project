#!/usr/bin/env bash
set -uo pipefail

echo "Starting overnight all-pairs runs..."
date

mkdir -p outputs/night_logs

SUMMARY_LOG="outputs/night_logs/overnight_summary_$(date +%Y%m%d_%H%M%S).log"
touch "$SUMMARY_LOG"

run_and_log () {
  NAME="$1"
  shift

  echo ""
  echo "============================================================"
  echo "Running: $NAME"
  echo "Command: $*"
  echo "Started at: $(date)"
  echo "============================================================"

  "$@" 2>&1 | tee "outputs/night_logs/${NAME}.log"
  STATUS=${PIPESTATUS[0]}

  if [ "$STATUS" -eq 0 ]; then
    echo "SUCCESS: $NAME finished at $(date)" | tee -a "$SUMMARY_LOG"
  else
    echo "FAILED: $NAME failed with exit code $STATUS at $(date)" | tee -a "$SUMMARY_LOG"
  fi

  return 0
}

# 1. All-pairs baseline with final reportable strategy: OW_transient_proxy
run_and_log "full_allpairs_baseline_proxy" \
  python -m src.run_integrated_rolling_simulations \
    --mode all_pairs \
    --scenario baseline \
    --strategy-model OW_transient_proxy \
    --experiment-name full_allpairs_baseline_proxy

# 2. All-pairs wrong-model stress
run_and_log "full_allpairs_wrong_model_proxy" \
  python -m src.run_integrated_rolling_simulations \
    --mode all_pairs \
    --scenario wrong_model \
    --wrong-model-mode both \
    --strategy-model OW_transient_proxy \
    --experiment-name full_allpairs_wrong_model_proxy

# 3. All-pairs signal-delay stress
run_and_log "full_allpairs_signal_delay_proxy" \
  python -m src.run_integrated_rolling_simulations \
    --mode all_pairs \
    --scenario signal_delay \
    --strategy-model OW_transient_proxy \
    --experiment-name full_allpairs_signal_delay_proxy

# 4. All-pairs forced-liquidation stress
# Runs deterministic_daily and probabilistic_daily p=10%, each with hard_block and capped_with_residual.
run_and_log "full_allpairs_forced_liq_both_proxy" \
  python -m src.run_integrated_rolling_simulations \
    --mode all_pairs \
    --scenario forced_liquidation \
    --strategy-model OW_transient_proxy \
    --liquidation-trigger-mode both \
    --liquidation-probability 0.10 \
    --liquidation-random-seed 42 \
    --liquidation-mode both \
    --experiment-name full_allpairs_forced_liq_both_proxy

# 5. All-pairs sizing sensitivity
run_and_log "full_allpairs_sizing_sensitivity_proxy" \
  python -m src.run_integrated_rolling_simulations \
    --mode all_pairs \
    --scenario sizing_sensitivity \
    --strategy-model OW_transient_proxy \
    --experiment-name full_allpairs_sizing_sensitivity_proxy

echo ""
echo "All overnight all-pairs runs attempted."
echo "Summary log: $SUMMARY_LOG"
cat "$SUMMARY_LOG"
date