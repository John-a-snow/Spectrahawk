from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class SimConfig:
    slots: int = 500
    bands: int = 32
    periodic_period: int = 37
    periodic_duty: float = 0.18
    seed: int = 7

@dataclass
class Episode:
    truth: np.ndarray
    visits: np.ndarray
    detected: np.ndarray
    rewards: np.ndarray
    selected: np.ndarray


def simulate(cfg: SimConfig, agent: str = "SpectraHawk-TS") -> Episode:
    """Run a deterministic educational RF scanning simulation."""
    if cfg.bands < 4 or cfg.slots < 1:
        raise ValueError("bands must be >= 4 and slots must be >= 1")
    rng = np.random.default_rng(cfg.seed)
    truth = np.zeros((cfg.slots, cfg.bands), dtype=np.uint8)
    fixed = [b for b in (2, 11, 23) if b < cfg.bands]
    for b in fixed:
        truth[:, b] = 1
    pband = min(17, cfg.bands - 1)
    phase = 5
    for t in range(cfg.slots):
        if ((t - phase) % cfg.periodic_period) < max(1, int(cfg.periodic_period * cfg.periodic_duty)):
            truth[t, pband] = 1
        # Frequency-agile emitter with a deterministic learnable hop pattern.
        agile_band = 5 + ((t // 4) * 7) % max(1, min(9, cfg.bands - 5))
        truth[t, agile_band] = 1
    selected = np.zeros(cfg.slots, dtype=int)
    detected = np.zeros(cfg.slots, dtype=bool)
    rewards = np.zeros(cfg.slots, dtype=float)
    last_seen = np.full(cfg.bands, -10_000, dtype=int)
    visits = np.zeros((cfg.slots, cfg.bands), dtype=np.uint8)
    counts = np.zeros(cfg.bands, dtype=float)
    age = np.zeros(cfg.bands, dtype=int)
    previous = 0
    for t in range(cfg.slots):
        if agent == "LinearSweep":
            b = t % cfg.bands
        elif agent == "Random":
            b = int(rng.integers(cfg.bands))
        else:
            # Closed-loop heuristic: Thompson-like posterior sample + age bonus.
            alpha = 1.0 + counts
            beta = 1.0 + np.maximum(0.0, (t + 1) / cfg.bands - counts)
            sample = rng.beta(alpha, beta)
            age_bonus = np.minimum(age, 4 * cfg.bands) / (4 * cfg.bands)
            stale = age >= 4 * cfg.bands
            score = sample + 0.65 * age_bonus
            if np.any(stale):
                b = int(np.flatnonzero(stale)[0])
            else:
                b = int(np.argmax(score))
        selected[t] = b
        visits[t, b] = 1
        active = bool(truth[t, b])
        p_det = 0.90 if active else 0.001
        hit = bool(rng.random() < p_det)
        detected[t] = hit
        novelty = active and (t - last_seen[b] > 50)
        if hit and active:
            counts[b] += 1
            last_seen[b] = t
            rewards[t] += 10.0 if b == pband and novelty else 3.0 if novelty else 1.0
        elif hit:
            rewards[t] -= 1.0
        elif last_seen[b] >= 0:
            rewards[t] -= 0.2
        rewards[t] -= 0.05 * abs(b - previous) / cfg.bands
        previous = b
        age += 1
        age[b] = 0
    return Episode(truth, visits, detected, rewards, selected)


def metrics(ep: Episode) -> dict[str, float]:
    active = ep.truth[ep.visits.astype(bool)].astype(bool)
    hits = ep.detected
    active_count = int(active.sum())
    inactive_count = len(active) - active_count
    pd = float((hits & active).sum() / active_count) if active_count else 0.0
    pfa = float((hits & ~active).sum() / inactive_count) if inactive_count else 0.0
    return {
        "Pd": pd,
        "Pfa": pfa,
        "Intercept rate / 1000": float((hits & active).sum() / len(ep.selected) * 1000),
        "Reward / slot": float(ep.rewards.mean()),
        "Cumulative reward": float(ep.rewards.sum()),
        "Band coverage": float((ep.visits.sum(axis=0) > 0).mean()),
    }
