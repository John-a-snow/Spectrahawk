import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
from spectrahawk.simulator import SimConfig, metrics, simulate

def test_deterministic():
    a = simulate(SimConfig(slots=100, bands=16, seed=3))
    b = simulate(SimConfig(slots=100, bands=16, seed=3))
    assert (a.truth == b.truth).all()
    assert (a.selected == b.selected).all()
    assert (a.detected == b.detected).all()

def test_agents_produce_valid_actions():
    for name in ["LinearSweep", "Random", "SpectraHawk-TS"]:
        ep = simulate(SimConfig(slots=120, bands=16, seed=9), name)
        assert ep.selected.min() >= 0
        assert ep.selected.max() < 16
        assert set(metrics(ep)) >= {"Pd", "Pfa", "Cumulative reward"}
