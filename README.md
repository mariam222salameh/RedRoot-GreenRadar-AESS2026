# RedRoot: Sustainable Cognitive Radar for Flood Detection
### IEEE AESS Sustainability Hackathon 2026 | Challenge 2

---

## 🌟 Overview

RedRoot is an intelligent, low-power radar monitoring system designed for autonomous flood detection. This project demonstrates a high-fidelity simulation of a cognitive radar system that optimizes energy consumption and spectrum usage through real-time environmental awareness.

This repository provides a verifiable simulation environment that validates the core logic of the RedRoot system, proving significant sustainability gains compared to conventional, always-on radar architectures.

---

## 🚀 Key Features

- **Cognitive Processing** — a System Intelligence Score (SIS) engine that adapts radar parameters dynamically based on environmental data:

  ```
  SIS = 0.35 × SNS + 0.35 × Anomaly + 0.20 × Spectrum + 0.10 × CDDnorm
  ```

- **Sustainability Focus** — drastically reduces energy consumption through intelligent duty-cycling and avoids congested frequency bands.
- **Validated Simulation** — a 24-hour flood event simulation serves as the primary proof-of-concept for the project's energy and carbon-saving claims.
- **Data-Driven Insights** — a comprehensive suite of analytical plots covering range profiles, power consumption, spectrum avoidance, and carbon accountability.

---

## 📊 Performance Metrics (Verified 24h Simulation)

The following results are derived from execution of the provided `redroot_simulation.py` script:

- **Energy Savings:** 87.7% reduction vs. conventional always-on radar (5,822 J vs. 47.5 kJ per day)
- **Carbon Accountability:** 8× reduction in Carbon Debt per Detection — 34.25 g CO₂/detection (RedRoot) vs. 279.60 g CO₂/detection (naive), based on **11 confirmed flood detections** in the 24h trace
- **Spectrum Efficiency:** 100% avoidance rate during simulated congested intervals

> CDD figures are only meaningful alongside a detection count — a system that saves energy by missing floods drives CDD toward infinity. All CDD claims above are reported against the same 11 detections used to compute both baselines.

**Note on Project Scope:** The performance metrics above are based on the validated 24-hour simulation code provided in this repository. Additional 11-day performance projections are documented in the project's Design Feasibility & Literature Grounding Report as theoretical design targets for future hardware/TinyML deployment and are **not** measured results of the current codebase.

---

## 🛠 Tech Stack

- **Language:** Python
- **Core Libraries:** numpy, scipy, matplotlib
- **Methodology:** High-fidelity radar signal modeling and energy ledger simulation

---

## 🔗 Project Resources & Documentation

- 🎥 **Project Demo Video:** https://drive.google.com/file/d/10Z477EYl-DiXEczLOXlsybEVnodKPeiB/view?usp=sharing
  Demonstrates execution of the validated 24-hour simulation (`redroot_simulation.py`) and generation of the performance analytical plots.
- 🌐 **Live Demo Dashboard:** https://redroot-green-radar.lovable.app
  Reflects the same audited headline figures as this README (87.7% energy reduction, 8× CDD improvement, 100% spectrum avoidance). Two supplementary metrics on the dashboard — the TinyML per-inference energy figure and "Peak Flood Power Saved" — are still being reconciled against the simulation code and should not yet be treated as audited.
- 📄 **Design Feasibility & Literature Grounding Report:** https://drive.google.com/file/d/1T5TwfE90mhCwXgftHDWfBzxh1PtTu3qz/view
  Contains theoretical projections and design targets for an 11-day operation scenario — distinct from the validated 24-hour simulation results presented in this repository.

---

## AI usage disclosure

AI tools (Claude, Anthropic) were used to accelerate code architecture design, documentation structure, and visualization layout. All simulation parameters, physical assumptions, threshold values, and result interpretation were determined and validated by the RedRoot team. Every result in this repository is reproducible from first principles using the provided code.

---

## Team

**RedRoot** | IEEE AESS Sustainability Hackathon 2026 | Challenge 2
