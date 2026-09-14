"""Weather caches: ERA5 (``era5_h3/``) and ORAS5 (``oras5_h3/``).

The caches store environmental variables *by H3 cell and timestamp*, but the
exact on-disk layout was produced by an upstream pipeline. ``WeatherStore``
auto-detects the most common layouts:

1. **Tabular files** (``.parquet`` / ``.csv`` / ``.pkl`` holding a DataFrame)
   with an H3 column (``h3`` / ``cell`` / ``h3_index`` / ``hex``) and a time
   column (``time`` / ``timestamp`` / ``valid_time`` / ``date``); remaining
   numeric columns are the variables. One file per month/region/etc. is fine —
   files are scanned lazily and indexed by (year, month).
2. **Columnar parallel-list dict pickles**:
   ``{"cells": [...], "wind_speed": [...], ...}`` with the month encoded in
   the *filename* as a ``YYYYMM`` token (e.g. ``era5_h3_201607.pkl``).
3. **Nested dict pickles**: ``{cell: {timestamp: {var: value}}}`` or
   ``{(cell, timestamp): {var: value}}``.
4. **NPZ bundles** with arrays ``cells``, ``times`` and one array per variable.

Everything is funneled into a single monthly aggregate table
``(cell, year, month) -> {var: mean}`` which is what the tabular MDP consumes
(``features.monthly_aggregate: true``). Per-timestamp lookup is also exposed
for the step-level GCRL environment.

If your layout differs, run
``python -m arctic_meta_irl.data.weather --inspect DIR`` to see what the
auto-detection finds, and (if needed) add a branch to
``WeatherStore._load_file`` — that is the single extension point.
"""
from __future__ import annotations

import argparse
import pickle
import re
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ..utils.logging import get_logger

log = get_logger(__name__)

_H3_COLS = ("h3", "cell", "h3_index", "hex", "h3_cell", "index")
_TIME_COLS = ("time", "timestamp", "valid_time", "date", "datetime", "ts")
_CELL_KEYS = ("cells", "cell", "h3", "h3_cells", "hexes")
# matches a 6-digit YYYYMM token in a filename (era5_h3_201607.pkl -> 2016, 07)
_YYYYMM_RE = re.compile(r"(\d{4})(\d{2})")


def _find_col(cols: Iterable[str], candidates: tuple[str, ...]) -> str | None:
    lower = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _is_columnar(obj: dict) -> bool:
    """True for {"cells": [...], "var1": [...], ...} parallel-list layouts."""
    cell_key = next((k for k in obj if k.lower() in _CELL_KEYS), None)
    if cell_key is None:
        return False
    n = len(obj[cell_key]) if hasattr(obj[cell_key], "__len__") else -1
    if n < 0:
        return False
    # every other entry must be a same-length sequence (a variable column)
    others = [k for k in obj if k != cell_key]
    return bool(others) and all(
        hasattr(obj[k], "__len__") and len(obj[k]) == n for k in others
    )


def _columnar_to_df(obj: dict, path: Path) -> pd.DataFrame:
    """Flatten a columnar parallel-list dict into [h3, time, var...] rows.

    The timestamp is the month encoded in the FILENAME (YYYYMM); each cell gets
    one row stamped at the 1st of that month (the data is already a monthly mean).
    """
    cell_key = next(k for k in obj if k.lower() in _CELL_KEYS)
    m = _YYYYMM_RE.search(path.stem)
    if m is None:
        raise ValueError(f"no YYYYMM in filename {path.name!r} for columnar layout")
    year, month = int(m.group(1)), int(m.group(2))
    data = {"h3": [str(c) for c in obj[cell_key]]}
    data["time"] = pd.Timestamp(year=year, month=month, day=1, tz="UTC")
    for k in obj:
        if k != cell_key:
            data[k] = list(obj[k])
    return pd.DataFrame(data)


class WeatherStore:
    """Unified accessor over one weather cache directory."""

    def __init__(self, root: str | Path, variables: list[str] | None = None,
                 name: str = "weather"):
        self.root = Path(root)
        self.name = name
        self.variables = variables  # None -> keep everything numeric
        self._monthly: dict[tuple[str, int, int], dict[str, float]] = {}
        self._loaded = False

    # ------------------------------------------------------------------ load
    def load(self) -> "WeatherStore":
        """Scan the cache dir once and build the monthly aggregate (idempotent)."""
        if self._loaded:
            return self
        if not self.root.exists():
            log.warning("[%s] cache dir %s does not exist; weather features will "
                        "be zero-filled.", self.name, self.root)
            self._loaded = True
            return self
        files = sorted(p for p in self.root.rglob("*")
                       if p.suffix.lower() in {".parquet", ".pq", ".csv",
                                               ".pkl", ".pickle", ".npz"})
        if not files:
            log.warning("[%s] no cache files found under %s", self.name, self.root)
        acc: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list))
        for p in files:
            try:
                df = self._load_file(p)
            except Exception as e:  # keep going; report at the end
                log.warning("[%s] could not parse %s (%s) — skipped", self.name, p.name, e)
                continue
            if df is None or df.empty:
                continue
            self._accumulate(df, acc)
        self._monthly = {
            k: {var: float(np.mean(vals)) for var, vals in d.items()}
            for k, d in acc.items()
        }
        log.info("[%s] monthly aggregate built: %d (cell,year,month) keys, vars=%s",
                 self.name, len(self._monthly), self.var_names())
        self._loaded = True
        return self

    def _load_file(self, p: Path) -> pd.DataFrame | None:
        """Parse one cache file into a long DataFrame [h3, time, var1, var2, ...].

        EXTENSION POINT: add custom layouts here.
        """
        suf = p.suffix.lower()
        if suf in {".parquet", ".pq"}:
            df = pd.read_parquet(p)
        elif suf == ".csv":
            df = pd.read_csv(p)
        elif suf == ".npz":
            z = np.load(p, allow_pickle=True)
            keys = set(z.files)
            if not {"cells", "times"} <= keys:
                raise ValueError("npz missing 'cells'/'times' arrays")
            data = {"h3": z["cells"].astype(str), "time": z["times"]}
            for k in keys - {"cells", "times"}:
                data[k] = z[k]
            df = pd.DataFrame(data)
        else:  # pickle: DataFrame, columnar parallel-list dict, or nested dicts
            with open(p, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, pd.DataFrame):
                df = obj.reset_index()
            elif isinstance(obj, dict) and _is_columnar(obj):
                # Columnar parallel-list layout: {"cells": [...], "var1": [...], ...}.
                # No per-row timestamp; the month is encoded in the FILENAME
                # (e.g. era5_h3_201607.pkl -> 2016-07). One row per cell.
                df = _columnar_to_df(obj, p)
            elif isinstance(obj, dict):
                rows = []
                for k, v in obj.items():
                    if isinstance(k, tuple) and len(k) == 2:        # (cell, ts) -> {var: val}
                        cell, ts = k
                        rows.append({"h3": str(cell), "time": ts, **dict(v)})
                    elif isinstance(v, dict):                       # cell -> ts -> {var: val}
                        for ts, vv in v.items():
                            row = {"h3": str(k), "time": ts}
                            row.update(vv if isinstance(vv, dict) else {"value": vv})
                            rows.append(row)
                df = pd.DataFrame(rows)
            else:
                raise ValueError(f"unsupported pickle payload: {type(obj)}")

        h3c = _find_col(df.columns, _H3_COLS)
        tc = _find_col(df.columns, _TIME_COLS)
        if h3c is None or tc is None:
            raise ValueError(f"no h3/time columns in {list(df.columns)[:8]}")
        df = df.rename(columns={h3c: "h3", tc: "time"})
        df["h3"] = df["h3"].astype(str)
        df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)
        return df.dropna(subset=["time"])

    def _accumulate(self, df: pd.DataFrame,
                    acc: dict[tuple[str, int, int], dict[str, list[float]]]) -> None:
        var_cols = [c for c in df.columns if c not in ("h3", "time")
                    and pd.api.types.is_numeric_dtype(df[c])]
        if self.variables is not None:
            var_cols = [c for c in var_cols if c in self.variables]
        if not var_cols:
            return
        df = df.assign(_y=df["time"].dt.year, _m=df["time"].dt.month)
        grouped = df.groupby(["h3", "_y", "_m"])[var_cols].mean()
        for (cell, y, m), row in grouped.iterrows():
            d = acc[(cell, int(y), int(m))]
            for var, val in row.items():
                if np.isfinite(val):
                    d[str(var)].append(float(val))

    # ----------------------------------------------------------------- query
    def var_names(self) -> list[str]:
        if self.variables is not None:
            return list(self.variables)
        names: set[str] = set()
        for d in self._monthly.values():
            names |= set(d)
        return sorted(names)

    def monthly(self, cell: str, year: int | None, month: int | None,
                fill: float = 0.0) -> dict[str, float]:
        """{var: mean value} for (cell, year, month); zero-filled when absent."""
        self.load()
        d = self._monthly.get((str(cell), int(year or 0), int(month or 0)), {})
        return {v: d.get(v, fill) for v in self.var_names()} if self.var_names() else dict(d)

    def monthly_vector(self, cell: str, year: int | None, month: int | None,
                       variables: list[str], fill: float = 0.0) -> np.ndarray:
        d = self.monthly(cell, year, month, fill=fill)
        return np.array([d.get(v, fill) for v in variables], dtype=np.float32)


# ------------------------------------------------------------------ CLI helper
def _inspect(root: str) -> None:
    store = WeatherStore(root).load()
    keys = list(store._monthly)[:5]
    print(f"dir            : {root}")
    print(f"monthly keys   : {len(store._monthly)}")
    print(f"variables      : {store.var_names()}")
    print(f"sample keys    : {keys}")
    for k in keys[:2]:
        print(f"  {k} -> {store._monthly[k]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Inspect a weather cache directory.")
    ap.add_argument("--inspect", required=True, help="cache dir (era5_h3 / oras5_h3)")
    _inspect(ap.parse_args().inspect)
