RedRoot: Sustainable Cognitive Radar for Flood Detection
IEEE AESS Sustainability Hackathon 2026 | Challenge 2

🌟 Overview
RedRoot is an intelligent, low-power radar monitoring system designed for autonomous flood detection. This project demonstrates a high-fidelity simulation of a cognitive radar system that optimizes energy consumption and spectrum usage through real-time environmental awareness.

This repository provides a verifiable simulation environment that validates the core logic of the RedRoot system, proving significant sustainability gains compared to conventional, always-on radar architectures.

🚀 Key Features
Cognitive Processing: Features a System Intelligence Score (SIS) engine that adapts radar parameters dynamically based on environmental data.

Sustainability Focus: Drastically reduces energy consumption through intelligent duty-cycling, avoiding congested frequency bands.

Validated Simulation: Includes a robust 24-hour flood event simulation that serves as the primary proof-of-concept for the project's energy and carbon-saving claims.

Data-Driven Insights: Generates a comprehensive suite of analytical plots covering range profiles, power consumption, spectrum avoidance, and carbon accountability.

📊 Performance Metrics (Verified 24h Simulation)
The following results are derived from the execution of the provided redroot_simulation.py script:

Energy Savings: 87.7% reduction compared to conventional always-on radar.

Carbon Accountability: 8× reduction in Carbon Debt per Detection (CDD).

Spectrum Efficiency: 100% avoidance rate during simulated congested intervals.

Note on Project Scope: The performance metrics above are based on the validated 24-hour simulation code provided in this repository. Additional 11-day performance projections are documented in the project's Design Feasibility & Literature Grounding Report as theoretical design targets for future hardware/TinyML deployment and are not measured results of the current codebase.

🛠 Tech Stack
Language: Python

Core Libraries: numpy, scipy, matplotlib

Methodology: High-fidelity radar signal modeling and energy ledger simulation.
