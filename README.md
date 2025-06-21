# MGAP - Markov Game Approximation

This project provides a framework for simulating and analyzing multi-agent reinforcement learning (MARL) in Markov games. It includes implementations of Q-learning agents, a Markov game environment simulator, and fluid approximation methods for theoretical analysis.

The project is not optimised.

## Features

- **Markov Game Environment**: Define state transitions, joint actions, and reward matrices.
- **Q-learning Agents**: Implement table-based Q-learning with epsilon-softmax policies.
- **Simulation Framework**: Run multiple independent simulations with logging capabilities.
- **Fluid Approximation**: Solve differential equations modeling learning dynamics at population scale.
- **Visualization Tools**: Compare simulation results with fluid approximations.

## Installation
```bash
pip install git+https://github.com/yannKerzreho/MarkovGameApproximation.git
```

## Project Structure

```
MarkovGameApproximation/
├── mgap/
│   ├── agents/
│   │   ├── reinforcer.py           # Base reinforcement learning agent
│   │   ├── qtable.py               # Q-learning implementations
│   ├── environments/
│   │   └── markov_game.py          # Markov game environment
│   └── solvers/
│       └── fluid_approximation.py  # ODE generator
│       └── simulator.py            # Multi-run simulation manager
```


## Example: Prisoner's Dilemma Evolution

Simulate learning dynamics in a 4-state prisoner's dilemma variant:

```python
import numpy as np
import mgap as mgap

def gain_matrix(g):
    """Prisoner's dilemma payoff matrix"""
    return np.array([
        [(2 * g, 2 * g), (g, 2 + g)],
        [(2 + g, g), (2, 2)]
    ])

# Game parameters
g = 1.5
states = 4
tau, epsilon, alpha, gamma = 0.5, 0.1, 0.01, 0.9
nb_iterations = 1000

# Initialize game environment
rewards_matrix = np.array([gain_matrix(g)] * states)
transition_matrix = np.array([[[[1, 0, 0, 0],
                                [0, 1, 0, 0]],
                               [[0, 0, 1, 0],
                                [0, 0, 0, 1]]]] * 4)

game = mgap.MarkovGame(state_space_size=4, 
                transition_matrix=transition_matrix,
                reward_matrix=rewards_matrix)

# Initialize agents
Q0 = np.array([[30.]*2]*4)
Q1 = np.array([[22., 20.]]*4)

reinforcer0 = mgap.QTableReinforcer(
    action_space_size=2, 
    state_space_size=4,
    alpha=alpha, gamma=gamma,
    tau=tau, epsilon=epsilon,
    initial_Q=Q0
)

reinforcer1 = mgap.QTableReinforcer(
    action_space_size=2,
    state_space_size=4,
    alpha=alpha, gamma=gamma,
    tau=tau, epsilon=epsilon,
    initial_Q=Q1
)

# Run simulations
print("Running simulations...")
simulator = mgap.Simulator()
simulator.run_simulations(game, [reinforcer0, reinforcer1], 
                        num_simulations=10, num_iterations=nb_iterations)

# Compute fluid approximation
print("Calculating fluid approximation...")
FA = mgap.FluidApproximation(game, [reinforcer0, reinforcer1])
t_span = (0, nb_iterations)
t_eval = np.linspace(0, nb_iterations, 100)

x_solution, S_solution = FA.solve_differential_system_invariant(
    [Q0.copy(), Q1.copy()],
    t_span,
    t_eval
)

# Visualize results
print("Generating visualizations...")
from mgap.environments.prisonnier_dilemma.utilities import nice_picture
nice_picture(simulator.final_log, [reinforcer0, reinforcer1], 
            x_solution, S_solution, t_eval)
```

## References

1. WIP

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.