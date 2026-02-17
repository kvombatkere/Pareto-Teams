# Computing Approximate Pareto Frontiers for Submodular Utility and Cost Tradeoffs

This repository provides a Python implementation of algorithms for approximating Pareto frontiers from the research paper **Computing Approximate Pareto Frontiers for Submodular Utility and Cost Tradeoffs**.


## Overview

In many data mining applications—including recommender systems, influence maximization, and team formation—the goal is to select a subset of elements (e.g., items, nodes, or experts) that maximizes a monotone submodular utility function while minimizing a cost function.
Classical approaches impose cardinality or knapsack constraints, or combine utility and cost into a single weighted objective. However, these approaches require fixing a tradeoff in advance and return only a single solution, providing limited insight into the space of achievable utility–cost tradeoffs.

In contrast, this work studies how to compute **representative sets of solutions** that expose different tradeoffs between utility and cost. We introduce $(\alpha_1,\alpha_2)$-approximate Pareto frontiers that provably approximate the achievable tradeoffs between submodular utility $f$ and cost $c$.
We formalize the **$\texttt{Pareto}$-$(f,c)$ problem** and develop efficient algorithms for multiple settings arising from different combinations of submodular utility and cost functions. Experiments on real-world datasets demonstrate that our algorithms efficiently compute high-quality approximate Pareto frontiers in practice.


## The $\texttt{Pareto}$-$(f,c)$ Problem

Let $\mathcal{V}$ be a ground set of $n$ items, a non-negative monotone submodular function  
$f : 2^{\mathcal{V}} \rightarrow \mathbb{R}_{\ge 0}$ to be maximized, and a non-negative cost function  
$c : 2^{\mathcal{V}} \rightarrow \mathbb{R}_{\ge 0}$ to be minimized.

The goal is to compute a polynomial-size set of solutions $\mathcal{S}' \subseteq 2^{\mathcal{V}}$ that forms an $(\alpha_1,\alpha_2)$-approximate Pareto frontier, meaning it approximately captures the optimal tradeoffs between utility $f$ and cost $c$.


## Real-World Applications

We map the $\texttt{Pareto}$-$(f,c)$ problem to real-world applications in team formation, influence maximization and  recommender systems; we provide definitions of $\mathcal{V}, f$, and $c$, for each case.

### Team Formation

Let $V$ be a set of experts and $U$ a universe of skills. Each expert $i \in V$ has skills $S_i \subseteq U$, and a task requires skills $T \subseteq U$.
The utility of a team $Q \subseteq V$ is the coverage function:

$$f(Q) = \left| \left( \bigcup_{i \in Q} S_i \right) \cap T \right|$$

To encode the linear cost function, we associate a hiring cost $w_i$ with each expert. To encode the diameter cost function we use a coordination graph, $G=(V,E)$ with edge weights $d(i,j)$ encoding pairwise communication costs between the experts.

---

### Recommender Systems

Let $V$ be a set of items with similarity matrix $M$, where $M(i,j)$ measures pairwise similarity.
The utility of a recommendation set $Q \subseteq V$ is the facility location function:

$$
f(Q) = \sum_{i \in V} \max_{j \in Q} M(i,j)
$$

This monotone submodular function captures representativeness of the selected items.

We represent the linear cost function $c(Q) = \sum_{i \in Q} w_i$,
where $w_i$ denotes the Euclidean distance of restaurant $i$ from the city center. 
The diameter cost follows from the graph $G=(V,E)$ with nodes characterized by their geographical coordinates and edge weights $d(i,j)$ corresponding to the geographical distances.

---

### Influence Maximization

Let $G = (V,E)$ be a social network and $Q \subseteq V$ a seed set.
The utility function is the expected influence spread:

$$f(Q) = \mathbb{E}[|\sigma(Q)|]$$

where $\sigma(Q)$ is the set of activated nodes under a stochastic diffusion model.

For linear costs, we associate a cost $w_i$ with each node $i \in V$, modeling the operational cost of targeting that individual (e.g., incentives or advertising spend), yielding $c(Q) = \sum_{i \in Q} w_i$.
To model coordination constraints among seeds, we define a graph
$G=(V,E)$ where edge weights $d(i,j)$ encode communication or geographical
distances between users, and instantiate the diameter cost as
$c(Q) = \max_{i,j \in Q} d(i,j)$, encouraging well-coordinated seed sets.


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