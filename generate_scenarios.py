#!/usr/bin/env python3
"""
Synthetic Drift Scenario Generator
- Generates fictional industries + KPI sets
- Produces synthetic time series with optional seasonality, drift, and shocks
- Creates "leading indicator" and "outcome" pairs with configurable lag
- Designed for explanatory visuals and narratives (values are arbitrary)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd


# -----------------------------
# Configuration dictionaries
# -----------------------------

INDUSTRIES: Dict[str, Dict] = {
    "E-commerce Fulfillment / 3PL": {
        "leading_indicators": [
            "Order cancellation rate (early drift)",
            "Carrier exception rate",
            "Pick-pack cycle time",
            "Late shipment risk index",
        ],
        "outcomes": [
            "Refund volume index (lagging)",
            "Repeat purchase proxy (lagging)",
            "Revenue leakage proxy (lagging)",
        ],
        "why_it_matters": [
            "Cancellations often precede margin erosion and support load.",
            "Carrier exceptions drift before visible SLA failures.",
            "Cycle time drift surfaces capacity and labor constraints early.",
        ],
    },
    "Subscription SaaS": {
        "leading_indicators": [
            "Support backlog index",
            "Time-to-first-value proxy",
            "Billing dispute index",
            "Feature adoption momentum",
        ],
        "outcomes": [
            "Logo churn proxy (lagging)",
            "Net revenue retention proxy (lagging)",
            "Expansion likelihood proxy (lagging)",
        ],
        "why_it_matters": [
            "Backlog drift can precede churn by weeks or months.",
            "TTFV drift often signals onboarding friction before renewals suffer.",
        ],
    },
    "Healthcare Clinic Network": {
        "leading_indicators": [
            "No-show rate",
            "Appointment lead time",
            "Claims rejection index",
            "Staffing coverage index",
        ],
        "outcomes": [
            "Collections delay proxy (lagging)",
            "Patient retention proxy (lagging)",
            "Cash strain proxy (lagging)",
        ],
        "why_it_matters": [
            "No-show drift reduces utilization before revenue shortfalls are obvious.",
            "Claims rejection drift precedes cash flow constraint visibility.",
        ],
    },
    "Manufacturing / Distribution": {
        "leading_indicators": [
            "Backorder index",
            "Supplier on-time delivery proxy",
            "Inventory accuracy proxy",
            "Scrap/rework momentum",
        ],
        "outcomes": [
            "Expedite cost proxy (lagging)",
            "OTIF proxy (lagging)",
            "Gross margin compression proxy (lagging)",
        ],
        "why_it_matters": [
            "Backorders drift before expedite costs and OTIF failures spike.",
            "Inventory accuracy drift drives downstream fulfillment errors.",
        ],
    },
}

SEVERITY_LABELS = ["IGNORE", "WATCH", "ACT"]

# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ScenarioMeta:
    scenario_id: str
    industry: str
    leading_indicator: str
    outcome_metric: str
    periods: int
    frequency: str
    lag_periods: int
    has_seasonality: bool
    drift_start: int
    drift_slope: float
    shock_period: Optional[int]
    shock_magnitude: float
    seed: int
    narrative: str


# -----------------------------
# Core generators
# -----------------------------

def _trend_component(t: np.ndarray, drift_start: int, drift_slope: float) -> np.ndarray:
    """Piecewise linear drift after drift_start."""
    drift = np.zeros_like(t, dtype=float)
    mask = t >= drift_start
    drift[mask] = (t[mask] - drift_start) * drift_slope
    return drift

def _seasonality_component(t: np.ndarray, amp: float, period: int) -> np.ndarray:
    """Simple sinusoidal seasonality."""
    return amp * np.sin(2 * math.pi * t / period)

def _shock_component(t: np.ndarray, shock_period: Optional[int], magnitude: float) -> np.ndarray:
    """Single-period shock (spike)."""
    if shock_period is None:
        return np.zeros_like(t, dtype=float)
    shock = np.zeros_like(t, dtype=float)
    if 0 <= shock_period < len(t):
        shock[shock_period] = magnitude
    return shock

def generate_timeseries(
    periods: int,
    base_level: float,
    noise_sigma: float,
    drift_start: int,
    drift_slope: float,
    has_seasonality: bool,
    seasonality_amp: float,
    seasonality_period: int,
    shock_period: Optional[int],
    shock_magnitude: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a synthetic series with drift, seasonality, shock, and noise."""
    t = np.arange(periods)
    trend = _trend_component(t, drift_start, drift_slope)
    seas = _seasonality_component(t, seasonality_amp, seasonality_period) if has_seasonality else 0.0
    shock = _shock_component(t, shock_period, shock_magnitude)
    noise = rng.normal(0.0, noise_sigma, size=periods)
    y = base_level + trend + seas + shock + noise

    # Ensure non-negative (many operational rates/indexes are >= 0)
    y = np.clip(y, a_min=0.0, a_max=None)
    return y

def make_lagged_outcome(
    leading: np.ndarray,
    lag_periods: int,
    outcome_noise_sigma: float,
    rng: np.random.Generator,
    scale: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    """Create an outcome series as a lagged, noisy function of the leading indicator."""
    shifted = np.roll(leading, lag_periods)
    shifted[:lag_periods] = shifted[lag_periods]  # pad start to avoid artificial zeros
    noise = rng.normal(0.0, outcome_noise_sigma, size=len(leading))
    outcome = bias + scale * shifted + noise
    return np.clip(outcome, a_min=0.0, a_max=None)

def classify_simple_signal(
    series: np.ndarray,
    drift_start: int,
    lookback: int = 6,
) -> Tuple[str, float, str]:
    """
    Very simple heuristic: compare mean of last lookback points to earlier baseline.
    Returns (severity, confidence, rationale).
    """
    n = len(series)
    lookback = min(lookback, n // 2 if n >= 2 else 1)
    recent = series[n - lookback :]
    baseline = series[max(0, drift_start - lookback) : drift_start] if drift_start > 0 else series[:lookback]

    # Guard: if baseline is empty, use early segment
    if len(baseline) == 0:
        baseline = series[:lookback]

    baseline_mean = float(np.mean(baseline))
    recent_mean = float(np.mean(recent))
    delta = recent_mean - baseline_mean

    # Confidence proxy based on effect size relative to recent variability
    denom = float(np.std(recent)) + 1e-6
    effect = delta / denom

    # Map to severity
    if effect < 0.6:
        severity = "IGNORE"
    elif effect < 1.4:
        severity = "WATCH"
    else:
        severity = "ACT"

    # Clamp confidence 0..1
    confidence = float(1.0 / (1.0 + math.exp(-effect)))  # sigmoid
    rationale = (
        f"Recent mean ({recent_mean:.1f}) vs baseline ({baseline_mean:.1f}) shows "
        f"{'increase' if delta >= 0 else 'decrease'}; effect size ~{effect:.2f}."
    )
    return severity, confidence, rationale

def craft_narrative(industry: str, leading: str, outcome: str, severity: str) -> str:
    """Generate a short operator-focused narrative explaining why the shape matters."""
    why_pool = INDUSTRIES[industry]["why_it_matters"]
    why = random.choice(why_pool)

    lead_vs_lag = (
        "This metric can behave as a leading signal when the drift persists across periods, "
        "but it becomes lagging when it spikes only after downstream impact is already visible."
    )

    action_line = {
        "IGNORE": "This pattern is likely noise; observe but avoid alerting operators.",
        "WATCH": "This pattern suggests emerging drift; validate context and watch for persistence.",
        "ACT": "This pattern is persistent and material; treat as actionable and investigate root cause drivers.",
    }[severity]

    return (
        f"{why} {lead_vs_lag} "
        f"Leading indicator: {leading}. Lagging outcome: {outcome}. {action_line}"
    )


# -----------------------------
# Scenario assembly
# -----------------------------

def build_scenario(periods: int, frequency: str, seed: int) -> Tuple[ScenarioMeta, pd.DataFrame]:
    random.seed(seed)
    rng = np.random.default_rng(seed)

    industry = random.choice(list(INDUSTRIES.keys()))
    leading = random.choice(INDUSTRIES[industry]["leading_indicators"])
    outcome = random.choice(INDUSTRIES[industry]["outcomes"])

    # Drift configuration (intentionally abstract)
    drift_start = random.randint(max(2, periods // 4), max(3, periods // 2))
    drift_slope = random.uniform(0.8, 3.5)  # arbitrary "index units per period"
    has_seasonality = random.random() < 0.35
    seasonality_amp = random.uniform(0.0, 6.0)
    seasonality_period = random.choice([6, 8, 12])

    shock_period = None
    shock_magnitude = 0.0
    if random.random() < 0.25:
        shock_period = random.randint(0, periods - 1)
        shock_magnitude = random.uniform(6.0, 18.0)

    lag_periods = random.choice([1, 2, 3])

    leading_series = generate_timeseries(
        periods=periods,
        base_level=random.uniform(20.0, 60.0),
        noise_sigma=random.uniform(1.0, 4.0),
        drift_start=drift_start,
        drift_slope=drift_slope,
        has_seasonality=has_seasonality,
        seasonality_amp=seasonality_amp,
        seasonality_period=seasonality_period,
        shock_period=shock_period,
        shock_magnitude=shock_magnitude,
        rng=rng,
    )

    outcome_series = make_lagged_outcome(
        leading=leading_series,
        lag_periods=lag_periods,
        outcome_noise_sigma=random.uniform(2.0, 6.0),
        rng=rng,
        scale=random.uniform(0.8, 1.3),
        bias=random.uniform(0.0, 8.0),
    )

    severity, confidence, rationale = classify_simple_signal(leading_series, drift_start=drift_start)
    narrative = craft_narrative(industry, leading, outcome, severity)

    scenario_id = f"{industry.lower().replace(' ', '_').replace('/', '-')}_{seed}"

    meta = ScenarioMeta(
        scenario_id=scenario_id,
        industry=industry,
        leading_indicator=leading,
        outcome_metric=outcome,
        periods=periods,
        frequency=frequency,
        lag_periods=lag_periods,
        has_seasonality=has_seasonality,
        drift_start=drift_start,
        drift_slope=float(drift_slope),
        shock_period=shock_period,
        shock_magnitude=float(shock_magnitude),
        seed=seed,
        narrative=narrative + " " + f"(Severity={severity}, Confidence={confidence:.2f}; {rationale})",
    )

    # Build long-format dataframe (easy to chart)
    df = pd.DataFrame({
        "scenario_id": [scenario_id] * periods * 2,
        "industry": [industry] * periods * 2,
        "metric_role": (["leading_indicator"] * periods) + (["lagging_outcome"] * periods),
        "metric_name": ([leading] * periods) + ([outcome] * periods),
        "period": list(range(1, periods + 1)) * 2,
        "value": np.concatenate([leading_series, outcome_series]),
    })

    return meta, df


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic drift scenarios (fictional data).")
    parser.add_argument("--scenarios", type=int, default=5, help="Number of scenarios to generate.")
    parser.add_argument("--periods", type=int, default=12, help="Number of periods per scenario.")
    parser.add_argument("--frequency", type=str, default="P", help="Abstract frequency label (e.g., P=period).")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument("--outdir", type=str, default="output", help="Output directory.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    metas: List[ScenarioMeta] = []
    frames: List[pd.DataFrame] = []

    for i in range(args.scenarios):
        meta, df = build_scenario(periods=args.periods, frequency=args.frequency, seed=args.seed + i)
        metas.append(meta)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    # Write outputs
    meta_path = os.path.join(args.outdir, "scenarios.json")
    data_path = os.path.join(args.outdir, "timeseries.csv")

    with open(meta_path, "w", encoding="utf-8") as f:
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "note": "All data is fictional/synthetic and intended for explanatory visuals only.",
            "scenarios": [asdict(m) for m in metas],
        }
        json.dump(payload, f, indent=2)

    all_df.to_csv(data_path, index=False)

    print(f"Wrote:\n- {meta_path}\n- {data_path}\n\nTip: Chart 'leading_indicator' only for drift visuals.")


if __name__ == "__main__":
    main()
