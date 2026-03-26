# ACO Algorithm for VRP

This folder contains an implementation of the Ant Colony Optimization (ACO) algorithm for solving the Capacitated Vehicle Routing Problem (CVRP).

## Overview

ACO is a metaheuristic inspired by the foraging behavior of ants. Ants deposit pheromone trails on paths, and the probability of choosing a path increases with the amount of pheromone. This implementation uses the basic ACO approach adapted for CVRP.

## Algorithm Details

### Key Components
- **Pheromone Matrix**: Represents the attractiveness of edges between nodes.
- **Heuristic Information**: Inverse of distance (1/distance).
- **Transition Probability**: P_ij = (τ_ij^α * η_ij^β) / Σ(τ_ik^α * η_ik^β), where τ is pheromone, η is heuristic.
- **Pheromone Update**:
  - Local: τ_ij = (1-ρ) * τ_ij + ρ * τ0
  - Global: τ_ij = (1-ρ) * τ_ij + ρ / L_best (for best path edges)

### Parameters
- `ants_num`: Number of ants (default: 20)
- `max_iter`: Maximum iterations (default: 100)
- `beta`: Heuristic importance (default: 2)
- `q0`: Exploitation probability (default: 0.1)
- `rho`: Evaporation rate (default: 0.1)

## Files

- `cvrp_base.py`: Graph representation and pheromone management
- `ant.py`: Ant class for path construction
- `basic_aco.py`: Main ACO algorithm
- `solver_aco.py`: Integration with project data and utilities

## How to Run

1. Ensure dependencies are installed:
   ```bash
   pip install pandas numpy matplotlib
   ```

2. Run the solver:
   ```bash
   python Algorithms/ACO/solver_aco.py
   ```

3. Results will be saved in `Results/` folder:
   - `aco_result.txt`: Route details
   - `aco_route_map.html`: Visualization

## Correlation with Theory

### ACO Fundamentals
- **Positive Feedback**: Good solutions get reinforced via pheromone increase.
- **Distributed Computation**: Multiple ants explore simultaneously.
- **Greedy Heuristic**: Distance-based heuristic guides search.

### CVRP Adaptation
- **Capacity Constraint**: Ants check vehicle capacity before moving.
- **Multiple Routes**: Depot returns represent vehicle trips.
- **Feasibility**: Only valid routes (capacity, connectivity) are accepted.

### Mathematical Formulation
- **Objective**: Minimize total distance ∑ d_ij for all edges in solution.
- **Constraints**:
  - Each customer visited exactly once.
  - Vehicle capacity not exceeded.
  - Routes start/end at depot.

## References

1. **Original ACO Paper**: Dorigo, M., & Gambardella, L. M. (1997). Ant colony system: a cooperative learning approach to the traveling salesman problem. IEEE Transactions on Evolutionary Computation, 1(1), 53-66.

2. **ACO for VRP**: Bullnheimer, B., Hartl, R. F., & Strauss, C. (1999). An improved ant system algorithm for the vehicle routing problem. Annals of Operations Research, 89, 319-328.

3. **MACS-VRPTW (Inspiration)**: Gambardella, L. M., Taillard, É., & Agazzi, G. (1999). MACS-VRPTW: A multiple ant colony system for vehicle routing problems with time windows. In New Ideas in Optimization (pp. 63-76). McGraw-Hill.

This implementation is based on the basic ACO framework, adapted for CVRP without time windows. The code follows the standard ACO steps: initialization, ant construction, pheromone update, and iteration until convergence or max iterations.