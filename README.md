# Arctic meta-IRL — code and data supplement

Anonymous supplement accompanying a paper submission. It contains the
implementation, configurations, trained checkpoints, and cached evaluation
outputs for a controlled comparison of inverse reinforcement learning (IRL)
reward models on Arctic vessel trajectories.

Start with [REPRODUCE.md](REPRODUCE.md): it maps each reported claim to the
artifact that backs it and states exactly what can and cannot be re-derived
here.

## Method

Navigation is modeled as a deterministic, goal-conditioned MDP on an H3
resolution-6 hexagonal graph of Arctic water cells (14,206 states, 39,053
edges), with an action mask removing unavailable moves at coastal and pentagon
cells. Three reward models are compared under one protocol:

- **MCE-IRL** — a shared linear reward fit by maximum-causal-entropy soft value
  iteration over a finite horizon.
- **AIRL** — a shared nonlinear reward learned adversarially against a masked
  PPO policy (`context_dim=0`), the capacity-matched control.
- **PEMIRL** — the same nonlinear reward conditioned on a per-vessel latent
  context, inferred by a bi-LSTM posterior with an info-max objective.

Additional baselines: canonical AIRL, GAIL, behavior cloning, and LSTM and
Transformer sequence policies. Each recovered reward is evaluated on held-out
action likelihood, decoded-route fidelity, and reward transfer (freeze the
reward, train a fresh PPO agent on it).

## Data

This archive contains **no AIS data** — no positions, tracks, vessel
identifiers, or per-voyage kinematics. Vessel identifiers in the cached
evaluation outputs are opaque pseudonyms (`V001`, `V002`, ...), stable across
files so per-vessel analyses group correctly. The underlying AIS corpus is
licensed by a third party and cannot be redistributed.

Models therefore cannot be retrained or re-evaluated from this archive. The
reported per-decision and per-vessel likelihoods, which are the primary
results, re-derive from the cached outputs with no data at all.

## Install

Python 3.12 recommended; `requirements.txt` pins the versions the reported
results were produced with. CPU is sufficient for everything runnable here.

```bash
python3.12 -m venv .venv && source .venv/bin/activate
make install
```

## What runs without data

```bash
make test     # pytest suite on synthetic fixtures (CPU, ~10 s)
make smoke    # full pipeline end-to-end at toy scale on synthetic data (CPU, ~1 min)
```

`make smoke` exercises MDP construction, dataset building, MCE-IRL, PEMIRL,
evaluation, and reward transfer on generated data, so the implementation can be
inspected end to end without the restricted corpus.

`make reproduce` and `make pipeline` require the AIS corpus plus ERA5 and ORAS5
fields and will not run here; `configs/` and `scripts/run_all.sh` document the
exact pipeline for anyone who obtains that data.

## Pipeline

```
scripts/00_build_mdp.py           gexf graph -> tabular MDP (bearing-sorted actions, masks)
scripts/01_build_dataset.py       voyages -> episodes; vessel-disjoint splits; TRAIN-only standardizer
scripts/02_heterogeneity.py       eta^2 and F-ratio effect-size analysis
scripts/03_train_mce_irl.py       shared linear reward via finite-horizon soft VI
scripts/04_train_pemirl.py        bi-LSTM posterior + AIRL + info-max + masked PPO
scripts/05_train_ppo_baseline.py  hand-crafted-cost PPO baseline
scripts/06_evaluate.py            LL, FEE, Hausdorff, overlap, length ratio -> results.json
scripts/07_generate_routes.py     decode O-D routes (support-set conditioning)
scripts/08_reward_transfer.py     freeze reward -> train fresh PPO on it -> goal-reach curve
scripts/09_significance.py        paired vessel-level significance tests
```

Each script takes `--config` (YAML with `base:` inheritance, see `configs/`),
`--seed`, and `--device`. `bash scripts/run_all.sh` chains them.

## Known gotcha: PEMIRL requires stability fixes

A naive PEMIRL port diverges (discriminator loss to ~1e11, policy collapse):
the info-max objective back-propagates into the discriminator through an
unbounded sum of f. `arctic_meta_irl/algos/pemirl.py` applies six fixes,
configured in `configs/pemirl.yaml`:

1. clamp |f| in the info-max target (`f_clamp: 10.0`)
2. standardize the info-max advantage per batch
3. gradient-norm clip the discriminator and posterior (`disc_grad_clip: 1.0`)
4. AIRL logit reward `r = f` (clamped, per-batch standardized) instead of
   saturating sigmoid(f) (`reward_logit`, `reward_norm`)
5. `info_coeff: 0.01` (down from 0.1)
6. delayed posterior training (`cnt_starting_iter: 10`)

Do not remove these if you retrain.

## Layout

```
arctic_meta_irl/     the package: data/ env/ features/ models/ algos/ eval/ utils/
configs/             default.yaml + per-method configs (YAML `base:` inheritance)
scripts/             numbered pipeline + run_all.sh + compare_results.py
tests/               CPU-only pytest suite with synthetic fixtures
smoke/               toy-scale end-to-end pipeline (make smoke)
notebooks/           analysis notebooks; see REPRODUCE.md for which run offline
models/              released trained checkpoints
results/reference/   reference values for the reported numbers
runs/eval/           cached per-episode and aggregate evaluation outputs
```

## License

MIT for code and released models. The AIS corpus is third-party-licensed and
not covered.
