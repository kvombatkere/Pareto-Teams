# Computing Approximate Pareto Frontiers for Submodular Utility and Cost Tradeoffs

This repository provides a Python implementation of algorithms for approximating Pareto frontiers for the research paper **omputing Approximate Pareto Frontiers for Submodular Utility and Cost Tradeoffs**. 

## Overview
In many data-mining applications, including recommender systems, influence maximization, and team formation, the goal is to pick
a subset of elements (e.g., items, nodes in a network, experts to perform a task) to maximize
a monotone submodular utility function while simultaneously minimizing a cost function.
Classical formulations model this tradeoff via cardinality or knapsack constraints, or by combining utility and cost into a single weighted objective. However, such approaches require committing to a specific tradeoff in advance and return only a single solution, offering limited insight into the space of viable utility–cost tradeoffs.
In this work, we depart from the single-solution paradigm and examine the problem of computing 
representative sets of high-quality solutions that expose different tradeoffs between submodular utility and cost.
For this, we introduce $(\alpha_1,\alpha_2)$-approximate Pareto
frontiers that provably approximate the achievable tradeoffs between submodular utility and cost.
Specifically, we formalize the $\texttt{Pareto}$-$\langle f,c \rangle$ problem and develop efficient algorithms for multiple instantiations arising from different combinations of submodular utility $f$ and cost functions $c$.
Our results offer a principled and practical framework for understanding and exploiting utility--cost tradeoffs in submodular optimization. 
Experiments on datasets from diverse application
domains demonstrate that our algorithms efficiently compute approximate Pareto frontiers in practice.

## The $\texttt{Pareto}$-$\langle f,c \rangle$ Problem
Consider a ground set $\mathcal{V}$ of $n$ items, a non-negative monotone submodular function $f:2^\mathcal{V}\to\mathbb{R}_{\ge 0}$ to be maximized, and a non-negative cost function $c:2^\mathcal{V}\to\mathbb{R}_{\ge 0}$ to be minimized.
The goal of $\texttt{Pareto}$-$\langle f,c \rangle$ is to find a polynomial-size set of solutions $\mathcal{S}' \subseteq 2^\mathcal{V}$ that forms an $(\alpha_1,\alpha_2) $-approximate Pareto frontier for the bi-objective $\langle f,c\rangle$ problem.

## Real-World Applications
We map the $\texttt{Pareto}$-$\langle f,c \rangle$ problem to real-world applications in team formation, influence maximization and  recommender systems; we provide definitions of $\mathcal{V}, f$, and $c$, for each case.

### Team Formation
Let $V$ be a set of experts and $U$ a universe of skills. Each expert $i \in V$ has skills $S_i \subseteq U$, and a task requires skills $T \subseteq U$. The utility of a team $Q \subseteq V$ is the monotone submodular coverage function  
$$
f(Q) = \left| \left( \bigcup_{i \in Q} S_i \right) \cap T \right|.
$$
Costs include linear hiring cost $c(Q) = \sum_{i \in Q} w_i$ and coordination cost defined as the diameter $c(Q) = \max_{i,j \in Q} d(i,j)$, where $d(i,j)$ measures communication or coordination distance.

---
### Recommender Systems
Let $V$ be a set of items with similarity matrix $M$, where $M(i,j)$ measures pairwise similarity. The utility of a recommendation set $Q \subseteq V$ is the facility location function  
$$
f(Q) = \sum_{i \in V} \max_{j \in Q} M(i,j),
$$
a monotone submodular function that captures representativeness. Costs include linear cost $c(Q) = \sum_{i \in Q} w_i$ and diameter cost $c(Q) = \max_{i,j \in Q} d(i,j)$, which encourages compact recommendations.

---
### Influence Maximization
Let $G=(V,E)$ be a social network and $Q \subseteq V$ a seed set. The utility function  
$$
f(Q) = \mathbb{E}[|\sigma(Q)|]
$$
measures the expected number of activated nodes under a stochastic diffusion model (e.g., Independent Cascade or Linear Threshold), and is monotone submodular. Costs include linear targeting cost $c(Q) = \sum_{i \in Q} w_i$ and diameter cost $c(Q) = \max_{i,j \in Q} d(i,j)$, which captures coordination or geographic constraints among selected seeds.


## Repository Structure
```
├── code/
    ├── pareto-cardinality/    # Cardinality Cost
        ├── paretoCardinality*.py
        ├── pareto-cardinality-*.ipynb

    ├── pareto-graph/          # Diameter Cost
        ├── paretoGraph*.py
        ├── pareto-graph-*.ipynb

    ├── pareto-knapsack/       # Knapsack Cost
        ├── paretoKnapsack*.py
        ├── pareto-knapsack-*.ipynb

├── datasets/                   
    ├── pickled_data/          # processed datasets
    ├── raw_data/  
    preprocess_*.ipynb         # pre-processing scripts

├── figures/                   # Saved figures for paper

├── README.md                  # This file
├── requirements.txt           # Package requirements
├── setup_env.sh               # Bash script to set up environment
├── utils.py                   # Python utilities
```

## Environment Setup
Run `./setup_env.sh` to create a new conda environment `pareto_env` with `Python 3.13` and install the required packages specified in `requirements.txt`.

## Citation