# INFOH410 Project

This repository compares three AI approaches on the same stochastic grid navigation problem:

- CSP replanning (Local Search and CSP topic)
- MDP value iteration
- Reinforcement Learning with tabular Q-learning

The task is to move from start to goal with stochastic motion, obstacles, and hazardous cells.

## Repository Layout

- src/environment.py: common stochastic environment
- src/instances.py: reproducible scenario generation
- src/csp_agent.py: CSP feasibility replanning with shortest-feasible path search
- src/mdp_agent.py: model-based value iteration
- src/rl_agent.py: model-free Q-learning
- src/evaluation.py: shared metrics
- run_experiments.py: runs all experiments and saves tables and plots
- report/main.tex: LaTeX report source

## Setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

   pip install -r requirements.txt

## Reproduce Results

Run:

python run_experiments.py

This creates:

- results/main_results.csv
- results/scalability_results.csv
- results/result_summary.txt
- results/main_success_rate.png
- results/main_avg_return.png
- results/scalability_runtime.png
- results/scalability_success.png