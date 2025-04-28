# Reinforcement Learning in Markov Games

This project provides a framework for simulating and analyzing multi-agent reinforcement learning (MARL) in Markov games. It includes implementations of Q-learning agents, a Markov game environment simulator, and fluid approximation methods for theoretical analysis.

## Features

- **Markov Game Environment**: Define state transitions, joint actions, and reward matrices.
- **Q-learning Agents**: Implement table-based Q-learning with epsilon-softmax policies.
- **Simulation Framework**: Run multiple independent simulations with logging capabilities.
- **Fluid Approximation**: Solve differential equations modeling learning dynamics at population scale.
- **Visualization Tools**: Compare simulation results with fluid approximations.

## Installation

1. Clone repository:
```bash
git clone https://github.com/yourusername/markov-game-rl.git
cd markov-game-rl
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

## Project Structure

```
markov-game-rl/
├── Class/
│   ├── Reinforcer.py        # Base reinforcement learning agent
│   ├── MarkovGame.py        # Markov game environment
│   ├── Simulator.py         # Multi-run simulation manager
│   ├── FluidApproximation.py # Differential equation solver
│   └── Qtable.py            # Q-learning implementations
├── Utilities.py             # Visualization and helper functions
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
from Class.MarkovGame import MarkovGame
from Class.Qtable import QTableReinforcer
from Class.Simulator import Simulator
from Class.FluidApproximation import FluidApproximation
from Class.Reinforcer import Q

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
from Utilities import nice_picture
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

![Example Visualization](https://via.placeholder.com/600x400?text=Sample+Output+Comparison)

## References

1. WIP

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.