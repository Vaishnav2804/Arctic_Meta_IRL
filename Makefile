.PHONY: install test smoke reproduce pipeline clean-smoke

# Install pinned dependencies and the package (editable).
install:
	pip install -r requirements.txt && pip install -e .

# CPU-only pytest suite with synthetic fixtures (~2 min).
test:
	pytest -q tests/

# Toy-scale end-to-end pipeline on synthetic data (CPU, minutes).
smoke: clean-smoke
	python smoke/make_data.py
	bash scripts/run_all.sh --config smoke/config.yaml

clean-smoke:
	rm -rf smoke/runs smoke/data

# Reproduce the paper numbers from the released checkpoints in models/
# (uses shipped caches in data/cache/; no training). Emits
# runs/eval/results.json + runs/ppo_on_pemirl_trajectory.png and diffs
# against results/reference/.
reproduce:
	python scripts/06_evaluate.py --config configs/pemirl.yaml --released
	python scripts/08_reward_transfer.py --config configs/pemirl.yaml --released --eval-only
	python scripts/compare_results.py

# Full retrain from raw data (GPU recommended for PEMIRL; hours).
pipeline:
	bash scripts/run_all.sh
