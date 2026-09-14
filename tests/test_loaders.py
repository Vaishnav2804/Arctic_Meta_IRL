"""Data loading: voyages pickle, registry JSON, GEXF graph, weather caches."""
import numpy as np

from arctic_meta_irl.data.loaders import (load_navigation_graph,
                                          load_vessel_registry, load_voyages)
from arctic_meta_irl.data.weather import WeatherStore


def test_load_voyages(cfg):
    voyages = load_voyages(cfg["paths"]["voyages_pkl"])
    assert len(voyages) == 30
    v = voyages[0]
    assert v.mmsi >= 316000000
    assert v.category in ("cargo", "tanker")
    assert len(v.cells) >= 3 and v.cells[0] == v.start and v.cells[-1] == v.goal


def test_category_filter(cfg):
    cargo = load_voyages(cfg["paths"]["voyages_pkl"], categories=("cargo",))
    assert all(v.category == "cargo" for v in cargo)
    assert 0 < len(cargo) < 30


def test_registry(cfg):
    reg = load_vessel_registry(cfg["paths"]["vessel_registry"])
    assert all(isinstance(k, int) for k in reg)
    rec = reg[316000000]
    assert rec["length"] > 0 and rec["type"] in ("cargo", "tanker")


def test_graph(cfg):
    g = load_navigation_graph(cfg["paths"]["graph_gexf"])
    assert g.number_of_nodes() == 40
    assert all("lat" in g.nodes[n] for n in g.nodes)


def test_weather_store(cfg):
    ws = WeatherStore(cfg["paths"]["era5_dir"], ["u10", "siconc"], "era5").load()
    assert set(ws.var_names()) >= {"u10", "siconc"}
    g = load_navigation_graph(cfg["paths"]["graph_gexf"])
    cell = sorted(g.nodes)[0]
    vec = ws.monthly_vector(cell, 2022, 7, ["u10", "siconc"])
    assert vec.shape == (2,) and np.isfinite(vec).all()
