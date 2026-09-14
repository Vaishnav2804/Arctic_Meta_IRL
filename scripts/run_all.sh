#!/usr/bin/env bash
# Full pipeline: MDP -> dataset -> heterogeneity -> training -> evaluation
#                -> reward transfer.
#
# Usage: bash scripts/run_all.sh [--config CFG] [--seed SEED]
#   --config CFG   use CFG for EVERY step (e.g. smoke/config.yaml);
#                  without it, the per-step configs are used
#                  (default.yaml for 00-02, mce_irl.yaml for 03,
#                   pemirl.yaml for 04/06/08, ppo_baseline.yaml for 05).
#   --seed SEED    RNG seed (default 0).
set -euo pipefail
cd "$(dirname "$0")/.."

CFG=""
SEED=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CFG="$2"; shift 2 ;;
    --seed)   SEED="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2
       echo "Usage: bash scripts/run_all.sh [--config CFG] [--seed SEED]" >&2
       exit 2 ;;
  esac
done

if [[ -n "$CFG" ]]; then
  python scripts/00_build_mdp.py        --config "$CFG"
  python scripts/01_build_dataset.py    --config "$CFG" --seed "$SEED"
  python scripts/02_heterogeneity.py    --config "$CFG"
  python scripts/03_train_mce_irl.py    --config "$CFG" --seed "$SEED"
  python scripts/04_train_pemirl.py     --config "$CFG" --seed "$SEED"
  python scripts/05_train_ppo_baseline.py --config "$CFG" --seed "$SEED"
  python scripts/06_evaluate.py         --config "$CFG" --seed "$SEED"
  echo "Skipping scripts/07_generate_routes.py (needs --mmsi/--start/--goal O-D arguments; run ad-hoc)."
  python scripts/08_reward_transfer.py  --config "$CFG" --seed "$SEED"
else
  python scripts/00_build_mdp.py        --config configs/default.yaml
  python scripts/01_build_dataset.py    --config configs/default.yaml --seed "$SEED"
  python scripts/02_heterogeneity.py    --config configs/default.yaml
  python scripts/03_train_mce_irl.py    --config configs/mce_irl.yaml      --seed "$SEED"
  python scripts/04_train_pemirl.py     --config configs/pemirl.yaml       --seed "$SEED" --n_traj 1000
  python scripts/05_train_ppo_baseline.py --config configs/ppo_baseline.yaml --seed "$SEED"
  python scripts/06_evaluate.py         --config configs/pemirl.yaml       --seed "$SEED"
  echo "Skipping scripts/07_generate_routes.py (needs --mmsi/--start/--goal O-D arguments; run ad-hoc)."
  python scripts/08_reward_transfer.py  --config configs/pemirl.yaml       --seed "$SEED"
fi

echo "Pipeline complete. See <workdir>/eval/results.json and <workdir>/ppo_on_pemirl_trajectory.png (workdir: runs/ by default, smoke/runs/ for --config smoke/config.yaml)"
