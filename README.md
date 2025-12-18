# Synthetic Drift Scenarios

This repository contains a small Python utility that generates **fictional,
synthetic drift scenarios** for explanatory and illustrative purposes.

The outputs are intended to demonstrate *how* early warning signals
can emerge before lagging outcomes — not to represent real operational data.

All values are synthetic.

---

## What this is (and is not)

**This is:**
- A lightweight scenario generator for illustrating drift, persistence, and lag
- Intended for visuals, narratives, and early-warning thinking
- Safe for public sharing (no real data, no proprietary logic)

**This is not:**
- A production analytics tool
- A forecasting model
- A representation of any real company or dataset

---

## Run in GitHub Codespaces (recommended)

Codespaces allows you to run this without any local setup.

### Steps

1. Open this repository on GitHub
2. Click **Code → Codespaces → Create codespace on main**
3. When the Codespaces environment opens, run the following in the terminal:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python generate_scenarios.py --scenarios 6 --periods 12 --seed 100 --outdir output

Outputs

After running, you will see:
	•	output/scenarios.json
Scenario metadata and short operator-focused narratives
	•	output/timeseries.csv
Long-format synthetic time series suitable for charting or visualization

You can open or download these files directly from the Codespaces file tree.

⸻

Common variations

Generate a single scenario for quick iteration:

python generate_scenarios.py --scenarios 1 --periods 8 --seed 101 --outdir output

Generate multiple scenarios with different shapes:

python generate_scenarios.py --scenarios 5 --periods 16 --seed 200 --outdir output


⸻

Notes on interpretation
	•	Periods are intentionally abstract (P1, P2, …), not calendar-based
	•	Numeric values are arbitrary and only meaningful in relative terms
	•	The goal is to reason about timing, drift, and persistence, not magnitude

⸻

License & usage

This code and its outputs are provided for illustrative purposes only.
Use freely, modify safely, and do not treat outputs as real-world indicators.

---

## Why this README works for you

- **Codespaces-first** → zero setup friction
- Keeps your **methodology-first framing**
- Explicitly guards against misuse or misinterpretation
- Reads well to:
  - operators
  - hiring managers
  - advisors
  - technically curious prospects

Most importantly: it sets expectations **correctly**.

---

## Optional (nice-to-have later)
If/when you add `/docs` with images:
- Add one line near the top:
  ```md
  See `/docs` for illustrative visuals generated from these scenarios.

No rush on that.


