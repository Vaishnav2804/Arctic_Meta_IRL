"""State features phi(s) for reward learning.

A feature vector is composed of (configurable via ``features:`` block):

* **Environmental** — ERA5 (wind u10/v10, waves swh/mwp, sst, sea ice
  siconc/sithick where present) and ORAS5 (votemper), aggregated monthly per
  cell by :class:`~arctic_meta_irl.data.weather.WeatherStore`.
* **Geometric** — normalized lat/lon; per-episode goal-relative terms:
  normalized great-circle distance-to-goal and cos/sin of the bearing toward
  the goal (so a *linear* MCE-IRL reward can express goal-seeking).
* **Vessel statics** — length/width/aspect + type one-hot (PEMIRL context input
  and the conditioned MDP of §II; for pooled MCE-IRL these are constant within
  an episode).

`FeatureBuilder.episode_matrix` produces the (T, D) matrix for one episode.
The static, episode-independent part `phi_static(s)` (env + lat/lon) can be
precomputed for all states for a given (year, month) — used by soft VI.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from ..data.vessel_features import VesselFeaturizer
from ..data.weather import WeatherStore
from ..utils.logging import get_logger

if TYPE_CHECKING:  # type-hint only; avoids a circular import with env
    from ..env.graph_mdp import GraphMDP

log = get_logger(__name__)


@dataclass
class FeatureSpec:
    era5_vars: list[str]
    oras5_vars: list[str]
    use_latlon: bool = True
    use_goal_distance: bool = True
    use_goal_bearing: bool = True
    use_vessel_static: bool = True
    standardize: bool = True

    @classmethod
    def from_config(cls, cfg) -> "FeatureSpec":
        f = cfg["features"]
        return cls(era5_vars=list(f.get("era5_vars", [])),
                   oras5_vars=list(f.get("oras5_vars", [])),
                   use_latlon=f.get("use_latlon", True),
                   use_goal_distance=f.get("use_goal_distance", True),
                   use_goal_bearing=f.get("use_goal_bearing", True),
                   use_vessel_static=f.get("use_vessel_static", True),
                   standardize=f.get("standardize", True))


class FeatureBuilder:
    def __init__(self, mdp: GraphMDP, era5: WeatherStore, oras5: WeatherStore,
                 vessels: VesselFeaturizer | None, spec: FeatureSpec):
        self.mdp = mdp
        self.era5 = era5.load()
        self.oras5 = oras5.load()
        self.vessels = vessels
        self.spec = spec
        # restrict to variables actually present in the caches
        self.era5_vars = [v for v in spec.era5_vars if v in set(self.era5.var_names())] \
            or list(spec.era5_vars)
        self.oras5_vars = [v for v in spec.oras5_vars if v in set(self.oras5.var_names())] \
            or list(spec.oras5_vars)
        # warn loudly if a configured weather var isn't in the cache — otherwise
        # it would be silently zero-filled and the reward would be blind to it.
        for tag, requested, store in (("era5", spec.era5_vars, self.era5),
                                      ("oras5", spec.oras5_vars, self.oras5)):
            missing = [v for v in requested if v not in set(store.var_names())]
            if missing:
                log.warning("[%s] requested vars %s not in cache (available: %s) "
                            "-> these features will be ZERO. Check config var names.",
                            tag, missing, store.var_names())
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._static_cache: dict[tuple[int, int], np.ndarray] = {}

    # ----------------------------------------------------------- dimensions
    @property
    def names(self) -> list[str]:
        n = [f"era5:{v}" for v in self.era5_vars] + [f"oras5:{v}" for v in self.oras5_vars]
        if self.spec.use_latlon:
            n += ["lat_norm", "lon_cos", "lon_sin"]
        if self.spec.use_goal_distance:
            n += ["goal_dist_norm"]
        if self.spec.use_goal_bearing:
            n += ["goal_bearing_cos", "goal_bearing_sin"]
        if self.spec.use_vessel_static and self.vessels is not None:
            n += self.vessels.names
        return n

    @property
    def dim(self) -> int:
        return len(self.names)

    @property
    def static_dim(self) -> int:
        """env + lat/lon block (episode-independent given year, month)."""
        d = len(self.era5_vars) + len(self.oras5_vars)
        if self.spec.use_latlon:
            d += 3
        return d

    # ------------------------------------------------------------- builders
    def phi_static_all(self, year: int | None, month: int | None) -> np.ndarray:
        """(S, static_dim) features for every state at a given (year, month)."""
        key = (int(year or 0), int(month or 0))
        if key in self._static_cache:
            return self._static_cache[key]
        S = self.mdp.n_states
        cols = []
        if self.era5_vars:
            era = np.stack([self.era5.monthly_vector(c, year, month, self.era5_vars)
                            for c in self.mdp.cells])
            cols.append(era)
        if self.oras5_vars:
            ora = np.stack([self.oras5.monthly_vector(c, year, month, self.oras5_vars)
                            for c in self.mdp.cells])
            cols.append(ora)
        if self.spec.use_latlon:
            lat = self.mdp.latlng[:, 0:1] / 90.0
            lon = np.radians(self.mdp.latlng[:, 1:2])
            cols.append(np.concatenate([lat, np.cos(lon), np.sin(lon)], axis=1))
        out = (np.concatenate(cols, axis=1) if cols
               else np.zeros((S, 0))).astype(np.float32)
        self._static_cache[key] = out
        return out

    def phi_goal(self, states: np.ndarray, goal: int) -> np.ndarray:
        """(N, goal_dim) goal-relative features for given state indices."""
        cols = []
        if self.spec.use_goal_distance:
            d = np.array([self.mdp.distance_km(int(s), goal) for s in states],
                         dtype=np.float32)
            cols.append((d / 1000.0)[:, None])          # ~Mm scale
        if self.spec.use_goal_bearing:
            b = np.array([self.mdp.bearing_to(int(s), goal) for s in states],
                         dtype=np.float32)
            cols.append(np.stack([np.cos(b), np.sin(b)], axis=1))
        return (np.concatenate(cols, axis=1) if cols
                else np.zeros((len(states), 0), dtype=np.float32))

    def phi_states(self, states: np.ndarray, goal: int, year: int | None,
                   month: int | None, mmsi: int | None = None) -> np.ndarray:
        """Full (N, D) feature matrix for arbitrary state indices."""
        static = self.phi_static_all(year, month)[states]
        parts = [static, self.phi_goal(states, goal)]
        if self.spec.use_vessel_static and self.vessels is not None:
            v = (self.vessels(mmsi) if mmsi is not None
                 else np.zeros(self.vessels.dim, dtype=np.float32))
            parts.append(np.tile(v, (len(states), 1)))
        out = np.concatenate(parts, axis=1).astype(np.float32)
        return self._standardize(out)

    def phi_all_states(self, goal: int, year: int | None, month: int | None,
                       mmsi: int | None = None) -> np.ndarray:
        """(S, D) features for *all* states — the soft-VI reward table input."""
        return self.phi_states(np.arange(self.mdp.n_states), goal, year, month, mmsi)

    def episode_matrix(self, state_seq: np.ndarray, goal: int, year: int | None,
                       month: int | None, mmsi: int | None) -> np.ndarray:
        return self.phi_states(np.asarray(state_seq), goal, year, month, mmsi)

    # -------------------------------------------------------- normalization
    # Per-feature normalization policy (not blanket z-scoring):
    #   - LOG1P features: heavily right-skewed, near-zero with rare spikes
    #     (ice). log1p compresses the tail BEFORE z-scoring so a single icy
    #     cell doesn't become a 12-sigma outlier that dominates a linear reward.
    #   - PASSTHROUGH features: already in a sensible bounded range with
    #     meaningful structure (cos/sin of lon & bearing; one-hot vessel types).
    #     Z-scoring these destroys the circular geometry and, for near-constant
    #     directions like lon_sin (sd~0.03), inflates pure noise ~30x.
    #   - everything else: standard z-score.
    _LOG1P_KEYS = ("ice_thickness", "ice_conc", "siconc", "sithick")
    _PASSTHROUGH_SUBSTR = ("lon_cos", "lon_sin", "bearing_cos", "bearing_sin",
                           "v_type=")

    def _feature_policy(self) -> tuple[np.ndarray, np.ndarray]:
        """Boolean masks (log1p, passthrough) aligned to self.names."""
        names = self.names
        log1p = np.array([any(k in n for k in self._LOG1P_KEYS) for n in names])
        passt = np.array([any(s in n for s in self._PASSTHROUGH_SUBSTR)
                          for n in names])
        return log1p, passt

    def _apply_log1p(self, X: np.ndarray, log1p_mask: np.ndarray) -> np.ndarray:
        if not log1p_mask.any():
            return X
        X = X.copy()
        # values are non-negative (ice), but guard against tiny negatives
        X[:, log1p_mask] = np.log1p(np.clip(X[:, log1p_mask], 0.0, None))
        return X

    def fit_standardizer(self, episode_matrices: list[np.ndarray]) -> None:
        if not self.spec.standardize or not episode_matrices:
            return
        X = np.concatenate(episode_matrices, axis=0)
        log1p_mask, passt_mask = self._feature_policy()
        self._log1p_mask, self._passt_mask = log1p_mask, passt_mask
        Xt = self._apply_log1p(X, log1p_mask)        # transform skewed cols first
        mu = Xt.mean(axis=0)
        sd = Xt.std(axis=0)
        sd[sd < 1e-6] = 1.0
        # passthrough features: identity (mu=0, sd=1) so geometry/one-hot survive
        mu[passt_mask] = 0.0
        sd[passt_mask] = 1.0
        self._mu, self._sd = mu, sd
        log.info("Feature standardizer fit on %d rows, D=%d "
                 "(log1p=%d, passthrough=%d, zscore=%d)",
                 len(X), X.shape[1], int(log1p_mask.sum()), int(passt_mask.sum()),
                 int((~log1p_mask & ~passt_mask).sum()))

    # safety clip on z-scored features: log1p reshapes the skewed-ice tail, but
    # the mass is so concentrated at zero that residual spikes can still hit
    # ~10 sigma. Clip the z-scored/log1p columns to +-CLIP so no single cell
    # dominates a linear reward. Passthrough features are exempt (already bounded).
    _CLIP = 5.0

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None or self._sd is None:
            return X
        log1p_mask = getattr(self, "_log1p_mask", None)
        if log1p_mask is not None:
            X = self._apply_log1p(X, log1p_mask)
        Z = (X - self._mu) / self._sd
        passt_mask = getattr(self, "_passt_mask", None)
        if passt_mask is None:
            return np.clip(Z, -self._CLIP, self._CLIP)
        clipped = np.clip(Z, -self._CLIP, self._CLIP)
        # keep passthrough columns exactly (don't clip bounded geometry/one-hot)
        clipped[:, passt_mask] = Z[:, passt_mask]
        return clipped

    def export_state(self) -> dict:
        return {"mu": self._mu, "sd": self._sd,
                "log1p_mask": getattr(self, "_log1p_mask", None),
                "passt_mask": getattr(self, "_passt_mask", None),
                "era5_vars": self.era5_vars, "oras5_vars": self.oras5_vars}

    def import_state(self, st: dict) -> None:
        self._mu, self._sd = st.get("mu"), st.get("sd")
        self._log1p_mask = st.get("log1p_mask")
        self._passt_mask = st.get("passt_mask")
