# Reproducing the reported numbers

## What this archive contains

Source code, configurations, trained model checkpoints, and cached evaluation
outputs. It contains **no AIS data of any kind** — no positions, no tracks, no
vessel identifiers, no per-voyage kinematics. Vessel identifiers in the cached
evaluation outputs have been replaced with opaque pseudonyms (`V001`, `V002`,
...); the mapping to real identifiers is not distributed. The pseudonyms are
stable across files, so per-vessel analyses still group correctly.

Consequently the models **cannot be retrained or re-evaluated** from this
archive. What can be re-derived, with no data at all, are the reported
per-decision and per-vessel likelihoods, which are the paper's primary results.

## Setup

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt && pip install -e .

## Runs with no data (verified)

    make test     # 43 tests, CPU, ~10 s
    make smoke    # toy end-to-end pipeline on synthetic data, ~1 min

`make smoke` exercises the full path — MDP build, dataset build, MCE-IRL,
PEMIRL, evaluation, and PPO reward transfer — on generated data, so the
implementation can be inspected end to end without the restricted corpus.

## Claim-to-artifact map

| Paper claim | Artifact | How to re-derive |
|---|---|---|
| Table 1, LL micro / macro, all models | `runs/eval/per_episode_ll*.json`, `runs/eval/robustness*.json` | `notebooks/03_per_vessel_ll.ipynb` |
| Worst-vessel MCE-IRL and AIRL likelihood | `runs/eval/per_episode_ll.json`, `runs/eval/per_episode_ll_noctx.json` | `notebooks/03_per_vessel_ll.ipynb` |
| Per-vessel win counts, 41 test vessels | `runs/eval/robustness.json` | `notebooks/03_per_vessel_ll.ipynb` |
| Support-size sweep | `runs/eval/support_sweep*.json`, `runs/eval/support_sweep_per_vessel*.csv` | read directly |
| Significance tests | `runs/eval/significance.json` | read directly |
| eta^2 by vessel / size class / subtype / category | `results/reference/eta_squared.json` | read directly |
| Decoded-route Hausdorff, overlap, length ratio, reach | `results/reference/routes_*.csv` | read directly |
| Reward transfer goal-reach | `models/ppo_on_pemirl/history.json`, `results/reference/results.json` | read directly |

## Aggregation convention (important)

Per-vessel likelihoods are aggregated over *query* episodes only: rows carrying
a score for every model being compared. In `runs/eval/per_episode_ll*.json` this
is the set where the model's LL field is present; the notebook applies it via
`dropna(subset=[SCHEME])`. Aggregating over all episodes instead yields
different worst-case vessels and does not reproduce Table 1's macro column. The
matched test set is 62,127 decisions across 41 vessels.

## Not reproducible from this archive

- Retraining any model, and `make reproduce`: both require the restricted AIS
  corpus plus ERA5/ORAS5 fields. `configs/` and `scripts/run_all.sh` document
  the exact pipeline; `make pipeline` runs it once that data is in place.
- The F-ratio heterogeneity statistics, which are computed from per-voyage
  descriptors (speed, path length, mean position). Those descriptors are AIS
  derivatives and are withheld. The eta^2 summary they accompany is included as
  an aggregate in `results/reference/eta_squared.json`.
- `notebooks/01_study_area_and_data.ipynb` and `notebooks/02_route_gallery.ipynb`
  are included to document the analysis, but require the restricted data to
  execute.

## Code layout

`arctic_meta_irl/` is the implementation; `configs/` holds one YAML per reported
model; `scripts/` holds the numbered pipeline stages driven by
`scripts/run_all.sh`; `tests/` and `smoke/` use synthetic fixtures only.
