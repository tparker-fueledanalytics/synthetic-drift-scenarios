#Fueled Analytics is a brand of Accelerato, LLC. © 2026 Accelerato, LLC. All rights reserved
#!/usr/bin/env python3
"""
Synthetic Drift Scenario Generator

Generates fictional synthetic metric-pair scenarios for explanatory visuals.

The generated values are arbitrary and do not represent real operational data,
validated indicators, forecasts, causal relationships, financial impact,
recommendations, alerts, monitoring logic, or production analytics logic.

This utility is intended only to create synthetic examples of:

- baseline-period values
- recent-period deviation
- persistence across abstract periods
- delayed comparison-series shapes

This is not the EWS engine.
This does not contain proprietary EWS logic.
This is not suitable for operational use.
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

SYNTHETIC_DOMAINS: Dict[str, Dict[str, List[str]]] = {
    "E-commerce Fulfillment / 3PL": {
        "input_metrics": [
            "Synthetic cancellation-rate index",
            "Synthetic carrier-exception index",
            "Synthetic cycle-time index",
            "Synthetic shipment-variance index",
        ],
        "comparison_metrics": [
            "Synthetic refund-volume comparison index",
            "Synthetic repeat-activity comparison index",
            "Synthetic leakage-style comparison index",
        ],
        "descriptor_notes": [
            "Included as a fictional operational-style label for illustration only.",
            "Used to demonstrate a synthetic deviation pattern across abstract periods.",
            "Included to support chart labels in synthetic examples.",
        ],
    },
    "Subscription SaaS": {
        "input_metrics": [
            "Synthetic support-backlog index",
            "Synthetic onboarding-time index",
            "Synthetic billing-dispute index",
            "Synthetic adoption-momentum index",
        ],
        "comparison_metrics": [
            "Synthetic churn-style comparison index",
            "Synthetic retention-style comparison index",
            "Synthetic expansion-style comparison index",
        ],
        "descriptor_notes": [
            "Included as a fictional operational-style label for illustration only.",
            "Used to demonstrate a synthetic deviation pattern across abstract periods.",
            "Included to support chart labels in synthetic examples.",
        ],
    },
    "Healthcare Clinic Network": {
        "input_metrics": [
            "Synthetic no-show-rate index",
            "Synthetic appointment-lead-time index",
            "Synthetic claims-rejection index",
            "Synthetic staffing-coverage index",
        ],
        "comparison_metrics": [
            "Synthetic collections-delay comparison index",
            "Synthetic retention-style comparison index",
            "Synthetic cash-strain-style comparison index",
        ],
        "descriptor_notes": [
            "Included as a fictional operational-style label for illustration only.",
            "Used to demonstrate a synthetic deviation pattern across abstract periods.",
            "Included to support chart labels in synthetic examples.",
        ],
    },
    "Manufacturing / Distribution": {
        "input_metrics": [
            "Synthetic backorder index",
            "Synthetic supplier-timing index",
            "Synthetic inventory-accuracy index",
            "Synthetic rework-momentum index",
        ],
        "comparison_metrics": [
            "Synthetic expedite-cost-style comparison index",
            "Synthetic fulfillment-style comparison index",
            "Synthetic margin-compression-style comparison index",
        ],
        "descriptor_notes": [
            "Included as a fictional operational-style label for illustration only.",
            "Used to demonstrate a synthetic deviation pattern across abstract periods.",
            "Included to support chart labels in synthetic examples.",
        ],
    },
}


CLASSIFICATION_LABELS = [
    "LOW_SYNTHETIC_DEVIATION",
    "MODERATE_SYNTHETIC_DEVIATION",
    "HIGH_SYNTHETIC_DEVIATION",
]


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ScenarioMeta:
    scenario_id: str
    synthetic_domain: str
    input_metric: str
    comparison_metric: str
    periods: int
    frequency: str
    delay_periods: int
    has_seasonality: bool
    drift_start: int
    drift_slope: float
    shock_period: Optional[int]
    shock_magnitude: float
    seed: int
    classification: str
    deviation_ratio: float
    arithmetic_rationale: str
    narrative: str


# -----------------------------
# Core generators
# -----------------------------

def _trend_component(t: np.ndarray, drift_start: int, drift_slope: float) -> np.ndarray:
    """
    Create a piecewise linear synthetic drift component after drift_start.

    This is an arithmetic shape generator only.
    It does not represent observed operational behavior.
    """
    drift = np.zeros_like(t, dtype=float)
    mask = t >= drift_start
    drift[mask] = (t[mask] - drift_start) * drift_slope
    return drift


def _seasonality_component(t: np.ndarray, amp: float, period: int) -> np.ndarray:
    """
    Create a simple sinusoidal synthetic seasonality component.

    This is included only to vary generated shapes.
    """
    return amp * np.sin(2 * math.pi * t / period)


def _shock_component(
    t: np.ndarray,
    shock_period: Optional[int],
    magnitude: float,
) -> np.ndarray:
    """
    Create a single-period synthetic spike.

    This is included only to vary generated shapes.
    """
    if shock_period is None:
        return np.zeros_like(t, dtype=float)

    shock = np.zeros_like(t, dtype=float)

    if 0 <= shock_period < len(t):
        shock[shock_period] = magnitude

    return shock


def generate_synthetic_series(
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
    """
    Generate a synthetic numeric series with optional drift, seasonality,
    single-period spike, and random noise.

    The returned values are arbitrary synthetic values.
    They are not real-world indicators.
    """
    t = np.arange(periods)

    trend = _trend_component(t, drift_start, drift_slope)
    seasonality = (
        _seasonality_component(t, seasonality_amp, seasonality_period)
        if has_seasonality
        else 0.0
    )
    shock = _shock_component(t, shock_period, shock_magnitude)
    noise = rng.normal(0.0, noise_sigma, size=periods)

    series = base_level + trend + seasonality + shock + noise

    # Keep values non-negative for cleaner chart examples.
    series = np.clip(series, a_min=0.0, a_max=None)

    return series


def make_delayed_comparison_series(
    input_series: np.ndarray,
    delay_periods: int,
    comparison_noise_sigma: float,
    rng: np.random.Generator,
    scale: float = 1.0,
    bias: float = 0.0,
) -> np.ndarray:
    """
    Create a delayed synthetic comparison series from the input series.

    This creates a mechanically related synthetic shape for illustration only.
    It does not represent an outcome, causal effect, forecast target,
    validation target, or operational consequence.
    """
    shifted = np.roll(input_series, delay_periods)

    # Pad the beginning to avoid artificial zero values.
    if delay_periods > 0:
        shifted[:delay_periods] = shifted[delay_periods]

    noise = rng.normal(0.0, comparison_noise_sigma, size=len(input_series))
    comparison = bias + scale * shifted + noise

    return np.clip(comparison, a_min=0.0, a_max=None)


def classify_synthetic_deviation(
    series: np.ndarray,
    drift_start: int,
    lookback: int = 6,
) -> Tuple[str, float, str]:
    """
    Compare the recent synthetic mean to an earlier synthetic baseline mean.

    Returns:
    - classification: deterministic synthetic deviation label
    - deviation_ratio: arithmetic ratio used only for illustrative classification
    - rationale: bounded arithmetic explanation

    This does not produce confidence, probability, recommendation, alert,
    forecast, root-cause inference, operational severity, or decision guidance.
    """
    n = len(series)

    if n == 0:
        raise ValueError("series must contain at least one value")

    lookback = min(lookback, n // 2 if n >= 2 else 1)

    recent = series[n - lookback :]

    if drift_start > 0:
        baseline = series[max(0, drift_start - lookback) : drift_start]
    else:
        baseline = series[:lookback]

    if len(baseline) == 0:
        baseline = series[:lookback]

    baseline_mean = float(np.mean(baseline))
    recent_mean = float(np.mean(recent))
    delta = recent_mean - baseline_mean

    denominator = float(np.std(recent)) + 1e-6
    deviation_ratio = float(delta / denominator)

    if deviation_ratio < 0.6:
        classification = "LOW_SYNTHETIC_DEVIATION"
    elif deviation_ratio < 1.4:
        classification = "MODERATE_SYNTHETIC_DEVIATION"
    else:
        classification = "HIGH_SYNTHETIC_DEVIATION"

    rationale = (
        f"Recent synthetic mean ({recent_mean:.1f}) compared with synthetic baseline "
        f"mean ({baseline_mean:.1f}); arithmetic deviation ratio={deviation_ratio:.2f}."
    )

    return classification, deviation_ratio, rationale


def craft_narrative(
    synthetic_domain: str,
    input_metric: str,
    comparison_metric: str,
    classification: str,
) -> str:
    """
    Generate a bounded explanatory note for synthetic charting examples.

    The narrative describes only the generated synthetic pattern.
    It does not state or imply forecasting, causality, recommendation,
    alerting, financial impact, root-cause inference, or operational action.
    """
    descriptor_note = random.choice(SYNTHETIC_DOMAINS[synthetic_domain]["descriptor_notes"])

    return (
        f"Synthetic domain label: {synthetic_domain}. "
        f"Synthetic input metric: {input_metric}. "
        f"Synthetic delayed comparison metric: {comparison_metric}. "
        f"Deterministic illustrative classification: {classification}. "
        f"{descriptor_note} "
        f"This narrative describes only the generated synthetic pattern and does not "
        f"state or imply forecasting, causality, recommendation, alerting, monitoring, "
        f"financial impact, root-cause inference, or operational action."
    )


# -----------------------------
# Scenario assembly
# -----------------------------

def build_scenario(
    periods: int,
    frequency: str,
    seed: int,
) -> Tuple[ScenarioMeta, pd.DataFrame]:
    """
    Build one synthetic scenario and return metadata plus long-format values.

    The output is suitable for public-safe demonstration artifacts when used
    with synthetic values only.
    """
    random.seed(seed)
    rng = np.random.default_rng(seed)

    synthetic_domain = random.choice(list(SYNTHETIC_DOMAINS.keys()))
    input_metric = random.choice(SYNTHETIC_DOMAINS[synthetic_domain]["input_metrics"])
    comparison_metric = random.choice(SYNTHETIC_DOMAINS[synthetic_domain]["comparison_metrics"])

    # Drift configuration is intentionally abstract and arbitrary.
    drift_start = random.randint(max(2, periods // 4), max(3, periods // 2))
    drift_slope = random.uniform(0.8, 3.5)

    has_seasonality = random.random() < 0.35
    seasonality_amp = random.uniform(0.0, 6.0)
    seasonality_period = random.choice([6, 8, 12])

    shock_period = None
    shock_magnitude = 0.0

    if random.random() < 0.25:
        shock_period = random.randint(0, periods - 1)
        shock_magnitude = random.uniform(6.0, 18.0)

    delay_periods = random.choice([1, 2, 3])

    input_series = generate_synthetic_series(
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

    comparison_series = make_delayed_comparison_series(
        input_series=input_series,
        delay_periods=delay_periods,
        comparison_noise_sigma=random.uniform(2.0, 6.0),
        rng=rng,
        scale=random.uniform(0.8, 1.3),
        bias=random.uniform(0.0, 8.0),
    )

    classification, deviation_ratio, arithmetic_rationale = classify_synthetic_deviation(
        input_series,
        drift_start=drift_start,
    )

    narrative = craft_narrative(
        synthetic_domain=synthetic_domain,
        input_metric=input_metric,
        comparison_metric=comparison_metric,
        classification=classification,
    )

    scenario_id = (
        f"{synthetic_domain.lower().replace(' ', '_').replace('/', '-')}_{seed}"
    )

    meta = ScenarioMeta(
        scenario_id=scenario_id,
        synthetic_domain=synthetic_domain,
        input_metric=input_metric,
        comparison_metric=comparison_metric,
        periods=periods,
        frequency=frequency,
        delay_periods=delay_periods,
        has_seasonality=has_seasonality,
        drift_start=drift_start,
        drift_slope=float(drift_slope),
        shock_period=shock_period,
        shock_magnitude=float(shock_magnitude),
        seed=seed,
        classification=classification,
        deviation_ratio=float(deviation_ratio),
        arithmetic_rationale=arithmetic_rationale,
        narrative=narrative,
    )

    df = pd.DataFrame(
        {
            "scenario_id": [scenario_id] * periods * 2,
            "synthetic_domain": [synthetic_domain] * periods * 2,
            "metric_role": (
                ["input_metric"] * periods
                + ["delayed_comparison_metric"] * periods
            ),
            "metric_name": (
                [input_metric] * periods
                + [comparison_metric] * periods
            ),
            "period": list(range(1, periods + 1)) * 2,
            "period_label": [f"{frequency}{i}" for i in range(1, periods + 1)] * 2,
            "value": np.concatenate([input_series, comparison_series]),
        }
    )

    return meta, df


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command-line arguments.

    This validation is for utility stability only.
    It is not analytical validation.
    """
    if args.scenarios < 1:
        raise ValueError("--scenarios must be at least 1")

    if args.periods < 4:
        raise ValueError("--periods must be at least 4")

    if not args.frequency:
        raise ValueError("--frequency must not be empty")

    if not args.outdir:
        raise ValueError("--outdir must not be empty")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate fictional synthetic drift scenarios for explanatory visuals. "
            "Outputs are not operational indicators."
        )
    )

    parser.add_argument(
        "--scenarios",
        type=int,
        default=5,
        help="Number of synthetic scenarios to generate.",
    )
    parser.add_argument(
        "--periods",
        type=int,
        default=12,
        help="Number of abstract periods per scenario. Minimum: 4.",
    )
    parser.add_argument(
        "--frequency",
        type=str,
        default="P",
        help="Abstract period label prefix, such as P for P1, P2, P3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for repeatable synthetic examples.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="output",
        help="Output directory.",
    )

    args = parser.parse_args()
    validate_args(args)

    os.makedirs(args.outdir, exist_ok=True)

    metas: List[ScenarioMeta] = []
    frames: List[pd.DataFrame] = []

    for i in range(args.scenarios):
        meta, df = build_scenario(
            periods=args.periods,
            frequency=args.frequency,
            seed=args.seed + i,
        )
        metas.append(meta)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)

    meta_path = os.path.join(args.outdir, "scenarios.json")
    data_path = os.path.join(args.outdir, "timeseries.csv")

    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "public_use_boundary": {
            "data_status": "fictional_synthetic_only",
            "not_real_operational_data": True,
            "not_ews_engine": True,
            "not_proprietary_ews_logic": True,
            "not_forecast": True,
            "not_recommendation": True,
            "not_alert": True,
            "not_monitoring": True,
            "not_root_cause_inference": True,
            "not_financial_impact_estimate": True,
            "not_production_analytics": True,
            "not_customer_or_pilot_evidence": True,
        },
        "note": (
            "All data is fictional and synthetic. Outputs are intended only for "
            "explanatory visuals and bounded documentation examples."
        ),
        "scenarios": [asdict(m) for m in metas],
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    all_df.to_csv(data_path, index=False)

    print(
        "Wrote:\n"
        f"- {meta_path}\n"
        f"- {data_path}\n\n"
        "Boundary: outputs are fictional synthetic examples only.\n"
        "Tip: chart 'input_metric' for a synthetic deviation visual."
    )


if __name__ == "__main__":
    main()
