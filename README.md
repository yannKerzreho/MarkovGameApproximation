# MGAP - Markov Game Approximation

This project provides a framework for simulating and analyzing multi-agent reinforcement learning (MARL) in Markov games. It includes implementations of Q-learning agents, a Markov game environment simulator, and fluid approximation methods for theoretical analysis.

The project is not optimised, mainly due to the non-paralleling of simulations and the use of a containing class to allow flexibility over the parameters of the algorithms used [np.Array or dictionary].

## Features

- **Markov Game Environment**: Define state transitions, joint actions, and reward matrices.
- **Q-learning Agents**: Implement table-based Q-learning with epsilon-softmax policies.
- **Simulation Framework**: Run multiple independent simulations with logging capabilities.
- **Fluid Approximation**: Solve differential equations modeling learning dynamics at population scale.
- **Visualization Tools**: Compare simulation results with fluid approximations.

## Installation

1. Clone repository:
```bash
git clone https://github.com/yannKerzreho/MarkovGameApproximation.git
cd MarkovGameApproximation
```

2. Install package:
```bash
pip install -e .
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

## Key Components

### 1. Markov Game Environment
Defines game dynamics through:
- State transition matrix
- Joint action reward matrix
- Multi-agent interaction logic

### 2. Q-learning Agents
Two implementations:
- **QTableReinforcer**: Standard Q-learning
- **QTableCounterFactualReinforcer**: Counterfactual updates

### 3. Fluid Approximation
Solves coupled differential equations representing:
- Q-value evolution
- State distribution dynamics

## Example: Prisoner's Dilemma Evolution

Simulate learning dynamics in a 4-state prisoner's dilemma variant:

```python
import numpy as np
from mgap.agents.qtable import QTableReinforcer
from mgap.agents.reinforcer import Q
from mgap.environments.markov_game import MarkovGame
from mgap.solvers.simulator import Simulator
from mgap.solvers.fluid_approximation import FluidApproximation

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

game = MarkovGame(state_space_size=4, 
                transition_matrix=transition_matrix,
                reward_matrix=rewards_matrix)

# Initialize agents
Q0 = Q(np.array([[30.]*2]*4))
Q1 = Q(np.array([[22., 20.]]*4))

reinforcer0 = QTableReinforcer(
    action_space_size=2, 
    state_space_size=4,
    alpha=alpha, gamma=gamma,
    tau=tau, epsilon=epsilon,
    initial_Q=Q0
)

reinforcer1 = QTableReinforcer(
    action_space_size=2,
    state_space_size=4,
    alpha=alpha, gamma=gamma,
    tau=tau, epsilon=epsilon,
    initial_Q=Q1
)

# Run simulations
print("Running simulations...")
simulator = Simulator()
simulator.run_simulations(game, [reinforcer0, reinforcer1], 
                        num_simulations=100, num_iterations=nb_iterations)

# Compute fluid approximation
print("Calculating fluid approximation...")
FA = FluidApproximation(game, [reinforcer0, reinforcer1])
t_span = (0, nb_iterations)
t_eval = np.linspace(0, nb_iterations, 1000)

x_solution, S_solution = FA.solve_differential_system_naive(
    [Q0.copy(), Q1.copy()],
    np.array([0.25]*4),
    t_span,
    t_eval
)

# Visualize results
print("Generating visualizations...")
from mgap.environments.prisonnier_dilemma.utilities import nice_picture
nice_picture(simulator.final_log, [reinforcer0, reinforcer1], 
            x_solution, S_solution, t_eval)
```

## Key Features Demonstrated

1. **Game Setup**:
   - 4-state prisoner's dilemma variant
   - State transitions based on previous joint actions
   - Parametric payoff matrix with cooperation incentive `g`

2. **Agent Configuration**:
   - Different initial Q-values for agents
   - Softmax exploration with temperature `tau`
   - Epsilon-greedy exploration rate

3. **Analysis Methods**:
   - Monte Carlo simulations with 10 runs
   - Differential equation-based fluid approximation
   - Comparative visualization of both approaches

## Output Interpretation

The `nice_picture` function generates:
1. Q-value evolution comparison
2. Policy probability trajectories
3. State distribution dynamics
4. Reward convergence patterns

## References

1. WIP

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.