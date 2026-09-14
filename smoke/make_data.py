"""Generate a tiny on-disk dataset mimicking the real input layout.

Weather caches use the REAL columnar layout: one pickle per month named with a
YYYYMM token (era5_h3_202207.pkl), holding {"cells": [...], "var1": [...], ...}
parallel lists (monthly means; the month lives in the filename).
"""
import json, pickle
from pathlib import Path
import networkx as nx
import numpy as np

root = Path("smoke/data"); root.mkdir(parents=True, exist_ok=True)
W, H = 8, 5
cell = lambda i, j: f"86cell{i:02d}{j:02d}fffff"
g = nx.Graph()
for i in range(W):
    for j in range(H):
        g.add_node(cell(i, j), lat=70 + .1 * j, lng=-95 + .2 * i)
for i in range(W):
    for j in range(H):
        if i + 1 < W: g.add_edge(cell(i, j), cell(i + 1, j))
        if j + 1 < H: g.add_edge(cell(i, j), cell(i, j + 1))
        if i + 1 < W and j + 1 < H: g.add_edge(cell(i, j), cell(i + 1, j + 1))
nx.write_gexf(g, root / "graph.gexf")

rng = np.random.default_rng(0)
mmsis = [316000000 + k for k in range(8)]
voyages = []
for v in range(60):
    m = mmsis[v % len(mmsis)]
    j = int(rng.integers(0, H)); L = int(rng.integers(5, W))
    cells = [cell(min(t, W - 1), j) for t in range(L + 1)]
    voyages.append(dict(mmsi=m, category="cargo" if v % 2 else "tanker",
                        cells=cells,
                        speeds=rng.uniform(5, 14, len(cells)).tolist(),
                        times=[f"2022-0{7 + v % 3}-0{1 + t % 9}" for t in range(len(cells))],
                        year=2022, month=7 + v % 3,
                        start=cells[0], goal=cells[-1], metadata={}))
pickle.dump(voyages, open(root / "voyages.pkl", "wb"))

json.dump({str(m): dict(length=120. + 25 * k, width=18. + 2 * k,
                        type="cargo" if k % 2 else "tanker", subtype=f"sub{k % 3}")
           for k, m in enumerate(mmsis)}, open(root / "registry.json", "w"))

cells_all = [str(c) for c in g.nodes]
for name, vars_ in (("era5_h3", ["u10", "v10", "siconc", "sithick"]),
                    ("oras5_h3", ["votemper"])):
    d = root / name; d.mkdir(parents=True, exist_ok=True)
    for mo in (7, 8, 9):
        obj = {"cells": cells_all}
        for var in vars_:
            obj[var] = (rng.uniform(0, 1.5, len(cells_all)).tolist()
                        if var in ("siconc", "sithick")     # ice: >=0, skewed
                        else rng.normal(size=len(cells_all)).tolist())
        pickle.dump(obj, open(d / f"{name}_2022{mo:02d}.pkl", "wb"))
print("smoke data written")
