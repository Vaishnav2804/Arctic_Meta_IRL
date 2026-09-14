"""Per-feature normalization policy of FeatureBuilder.

The builder does NOT blanket z-score:
  - ice-named features (ice_thickness / ice_conc / siconc / sithick) are
    log1p-transformed BEFORE z-scoring (skew compression);
  - cos/sin geometry (lon_cos/lon_sin, bearing_cos/bearing_sin) and vessel
    type one-hots (``v_type=``) are passthrough: mu=0, sd=1, never clipped;
  - every other feature is z-scored and clipped to +-5.0.
"""
import numpy as np

ICE_KEYS = ("ice_thickness", "ice_conc", "siconc", "sithick")
PASS_SUBSTR = ("lon_cos", "lon_sin", "bearing_cos", "bearing_sin", "v_type=")
CLIP = 5.0


def _expected_masks(fb):
    names = fb.names
    ice = np.array([any(k in n for k in ICE_KEYS) for n in names])
    passt = np.array([any(s in n for s in PASS_SUBSTR) for n in names])
    return names, ice, passt


def test_masks_match_naming_policy(feature_builder):
    fb = feature_builder
    names, ice, passt = _expected_masks(fb)
    assert fb._log1p_mask is not None and fb._passt_mask is not None
    assert (np.asarray(fb._log1p_mask) == ice).all()
    assert (np.asarray(fb._passt_mask) == passt).all()
    # fixture config: era5 siconc + sithick are the only ice-named features
    assert set(np.array(names)[ice]) == {"era5:siconc", "era5:sithick"}
    # lon cos/sin + goal-bearing cos/sin + at least 2 type one-hots
    assert {"lon_cos", "lon_sin", "goal_bearing_cos",
            "goal_bearing_sin"} <= set(np.array(names)[passt])
    assert sum(n.startswith("v_type=") for n in np.array(names)[passt]) >= 2
    assert not (ice & passt).any()


def test_passthrough_identity_mu_sd(feature_builder):
    fb = feature_builder
    _, _, passt = _expected_masks(fb)
    st = fb.export_state()
    assert np.allclose(st["mu"][passt], 0.0)
    assert np.allclose(st["sd"][passt], 1.0)
    # z-scored columns have genuine statistics (sd floored, never 0)
    assert (st["sd"] > 0).all()


def test_export_state_contents(feature_builder):
    fb = feature_builder
    st = fb.export_state()
    assert set(st) >= {"mu", "sd", "log1p_mask", "passt_mask",
                       "era5_vars", "oras5_vars"}
    assert st["mu"].shape == (fb.dim,) and st["sd"].shape == (fb.dim,)
    assert len(st["log1p_mask"]) == fb.dim and len(st["passt_mask"]) == fb.dim
    assert st["era5_vars"] == fb.era5_vars and st["oras5_vars"] == fb.oras5_vars


def test_log1p_applied_before_zscore(feature_builder):
    fb = feature_builder
    _, ice, passt = _expected_masks(fb)
    i_ice = int(np.flatnonzero(ice)[0])
    X = np.zeros((1, fb.dim), dtype=np.float32)
    X[0, i_ice] = 3.0
    out = fb._standardize(X.copy())
    expected = np.clip((np.log1p(3.0) - fb._mu[i_ice]) / fb._sd[i_ice],
                       -CLIP, CLIP)
    assert np.isclose(out[0, i_ice], expected, atol=1e-5)
    # negative ice values are floored at 0 before log1p
    Xn = np.zeros((1, fb.dim), dtype=np.float32)
    Xn[0, i_ice] = -1.0
    outn = fb._standardize(Xn.copy())
    expected0 = np.clip((0.0 - fb._mu[i_ice]) / fb._sd[i_ice], -CLIP, CLIP)
    assert np.isclose(outn[0, i_ice], expected0, atol=1e-5)


def test_zscore_columns_clipped_at_5(feature_builder):
    fb = feature_builder
    _, ice, passt = _expected_masks(fb)
    i_z = int(np.flatnonzero(~ice & ~passt)[0])  # plain z-scored column
    X = np.zeros((1, fb.dim), dtype=np.float32)
    X[0, i_z] = 1e9
    out = fb._standardize(X.copy())
    assert out[0, i_z] == CLIP
    X[0, i_z] = -1e9
    out = fb._standardize(X.copy())
    assert out[0, i_z] == -CLIP


def test_passthrough_unchanged_and_never_clipped(feature_builder):
    fb = feature_builder
    _, _, passt = _expected_masks(fb)
    i_p = int(np.flatnonzero(passt)[0])
    X = np.zeros((1, fb.dim), dtype=np.float32)
    X[0, i_p] = 0.25
    out = fb._standardize(X.copy())
    assert np.isclose(out[0, i_p], 0.25, atol=1e-6)  # identity transform
    X[0, i_p] = 7.5  # beyond the +-5 clip: passthrough must be exempt
    out = fb._standardize(X.copy())
    assert np.isclose(out[0, i_p], 7.5, atol=1e-6)


def test_episode_matrices_respect_clip(pipeline):
    fb, eps = pipeline
    _, _, passt = _expected_masks(fb)
    for ep in eps[:5]:
        phi = fb.episode_matrix(ep.states[:-1], ep.goal, ep.year, ep.month,
                                ep.mmsi)
        assert np.isfinite(phi).all()
        assert (np.abs(phi[:, ~passt]) <= CLIP + 1e-6).all()
