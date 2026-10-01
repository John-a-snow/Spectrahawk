# SpectraHawk

Simulation-only adaptive spectrum scanning prototype for Smart India Hackathon 2026, Team Que Sera Sera.

## Run

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit.

## Scope

This first runnable build provides a deterministic simulated RF environment, receiver dwell trail, LinearSweep/Random/SpectraHawk-TS policies, live heatmap, KPI cards, reward curve, and CSV export. It intentionally does not claim results from the complete research specification yet. The remaining research modules are to be implemented and benchmarked before any headline performance claim is made.
