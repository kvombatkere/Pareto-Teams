# Pareto-Submodular-Cost: Approximating Pareto Frontiers

This repository implements algorithms for approximating Pareto frontiers in problems with monotone submodular objectives and costs, applied to team formation, influence maximization, and recommendations.

## Problem Definition
Given a ground set \( V \) of items, a monotone submodular function \( f \) to maximize, and a cost function \( c \) to minimize, find an approximate Pareto set.

## Variants
- **Cardinality**: Enforces size limits.
- **Knapsack**: Budget constraints.
- **Diameter**: Graph-based costs.

## Applications
- **Team Formation**: Balance skill coverage and hiring/communication costs.
- **Influence Maximization**: Trade-off spread and seed costs.
- **Recommendations**: Quality vs. diversity/distance.

## Repository Structure
├── code/
    ├── pareto-cardinality/    # Cardinality cost variants
        ├── paretoCardinalityTeams.py
        ├── paretoCardinalityRestaurants.py
    ├── pareto-graph/          # Graph/diameter cost variants
        ├── paretoGraphTeams.py
    ├── pareto-knapsack/       # Knapsack cost variants
        ├── paretoKnapsackTeams.py
        ├── paretoKnapsackRestaurants.py
├── datasets/                  # Raw and processed datasets
├── README.md                  # This file
├── requirements.txt           # Package requirements
├── setup_env.sh               # Bash script to set up environment

## Setup
Run `./setup_env.sh` to create a conda environment with Python 3.13 and install dependencies.

## Citation
<!-- Add citation -->