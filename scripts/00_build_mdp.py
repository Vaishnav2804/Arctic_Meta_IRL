"""00 — Build the tabular MDP from the H3 res-6 navigation graph.

    python scripts/00_build_mdp.py --config configs/default.yaml

Reads  : paths.graph_gexf   (all_water_arctic_r6.gexf: 14,206 nodes, 39,053 edges)
Writes : paths.mdp_cache    (data/cache/mdp_r6.npz)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from common import base_parser, load_cfg, log

from arctic_meta_irl.data.loaders import load_navigation_graph
from arctic_meta_irl.env.graph_mdp import build_mdp


def main() -> None:
    args = base_parser(__doc__).parse_args()
    cfg = load_cfg(args)

    graph = load_navigation_graph(cfg["paths"]["graph_gexf"])
    log.info("Navigation graph: %d nodes, %d edges",
             graph.number_of_nodes(), graph.number_of_edges())

    mdp = build_mdp(graph,
                    allow_stay=bool(cfg["mdp"]["allow_stay"]),
                    max_actions=int(cfg["mdp"]["max_actions"]))

    deg = (mdp.next_state >= 0).sum(axis=1)
    log.info("MDP: |S|=%d, |A|=%d, mean valid actions=%.2f, "
             "isolated states=%d",
             mdp.n_states, mdp.n_actions, float(deg.mean()),
             int((deg <= 1).sum()))
    log.info("Mean hex step length: %.2f km",
             float(np.nanmean(np.where(mdp.step_km > 0, mdp.step_km, np.nan))))

    out = Path(cfg["paths"]["mdp_cache"])
    out.parent.mkdir(parents=True, exist_ok=True)
    mdp.save(out)
    log.info("MDP cached at %s", out)


if __name__ == "__main__":
    main()
