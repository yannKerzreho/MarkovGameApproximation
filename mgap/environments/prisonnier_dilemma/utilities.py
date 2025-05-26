import numpy as np
from mgap.agents.reinforcer import Q
from scipy.special import logsumexp

def moving_average(data, window_size=5):
    """
    Smooths the data using a moving average.
    
    :param data: Array-like, the data to smooth.
    :param window_size: Size of the moving average window.
    :return: Smoothed data as a NumPy array.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')


def SMEpolicy(Q: Q, current_state: int, tau: float, epsilon: float):
    """
    Implements a Softmax policy with exploration.

    Args:
        Q (Q): Q-table storing state-action values.
        current_state (int): The current state of the agent.
        tau (float): Temperature parameter controlling exploration (higher tau leads to more uniform probabilities).
        epsilon (float): Exploration probability for epsilon-greedy behavior.

    Returns:
        numpy.ndarray: Probabilities of selecting each action.
    """
    state_Q_values = Q.data[current_state]
    Q_scaled = state_Q_values / tau
    softmax = np.exp(Q_scaled - logsumexp(Q_scaled))
    return softmax * (1 - epsilon) + epsilon / len(softmax)  # Epsilon-greedy adjustment

def nice_picture(logs, reinforcers, x_solution, S_solution, t_eval, rescale=True):
    """
    Create a 2x2 plot layout comparing Q-values, probabilities, and state distributions.
    Compatible with Q instances from Simulator and FluidApproximation.

    :param logs: Logs from the simulator, containing mean_Q and state_proportions
    :param reinforcers: List of reinforcers (agents) for policy evaluation
    :param x_solution: List of lists of Q instances from solve_differential_system
    :param S_solution: State distributions from solve_differential_system
    :param t_eval: Time steps for the solutions
    :param rescale: If True, rescale x-axis between 0 and 1
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    def get_q_value(q_instance, state_id, action_id):
        """Helper function to get Q-value regardless of data structure"""
        if isinstance(q_instance.data, np.ndarray):
            return q_instance.data[state_id, action_id]
        else:
            return q_instance.data["Qvalues"][tuple(state_id, action_id)]

    # Rescaled x-axis
    if rescale:
        x_eval = np.linspace(0, 1, len(t_eval))
        x_logs = np.linspace(0, 1, len(logs['mean_Q'][0]))
        x_logs_state = np.linspace(0, 1, len(logs["state_proportions"][:, 0]))
    else:
        x_eval = t_eval
        x_logs = range(len(logs['mean_Q'][0]))
        x_logs_state = range(len(logs["state_proportions"][:, 0]))

    # Extract dimensions from first Q instance
    num_players = len(x_solution)
    first_Q = logs["mean_Q"][0][0]  # Get first Q instance from logs
    
    num_states, num_actions = 4, 2

    # Extract state and action space sizes based on Q type
    if isinstance(first_Q.data, np.ndarray):
        logs_mean_Q_temp = logs["mean_Q"].copy()
        x_solution_temp = x_solution
    else:
        # For dictionary Q structure, assume consistent shape across states
        x_solution_temp = [[], []]
        logs_mean_Q_temp = [[], []]

        for t in range(len(t_eval)):
            x_solution_temp[0].append(Q(np.array(x_solution[0][t].data['Qvalues'])))
            x_solution_temp[1].append(Q(np.array(x_solution[1][t].data['Qvalues'])))

        for t in range(len(logs['mean_Q'][0])):
            logs_mean_Q_temp[0].append(Q(np.array(logs['mean_Q'][0][t].data['Qvalues'])))
            logs_mean_Q_temp[1].append(Q(np.array(logs['mean_Q'][1][t].data['Qvalues']))) 
        
    # Define styles
    state_styles = ['#44AF69', '#2E86AB', '#C73E1D', '#F18F01', '#A23B72', '#3B1F2B', '#7768AE', '#1B998B', '#E15554', '#4D9DE0']
    action_styles = ['-', '--']
    player_styles = ['-', '--']

    states_names = ['CC', 'CD', 'DC', 'DD']
    actions_names = ['C', 'D']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot Q-values for Player 0
    for state_id in range(num_states):
        for action_id in range(num_actions):
            # Solution
            q_values_solution = [
                get_q_value(x_solution_temp[0][t], state_id, action_id)
                for t in range(len(t_eval))
            ]
            axes[0, 0].plot(
                x_eval, q_values_solution,
                linestyle=action_styles[action_id],
                color=state_styles[state_id % len(state_styles)],
                label=rf"$Q^0_{{{states_names[state_id]}, {actions_names[action_id]}}}$"
            )
            # Logs
            q_values_logs = [
                get_q_value(q, state_id, action_id)
                for q in logs_mean_Q_temp[0]
            ]
            axes[0, 0].plot(
                x_logs, q_values_logs,
                linestyle=action_styles[action_id],
                color=state_styles[state_id % len(state_styles)],
                alpha=0.4, linewidth=3
            )
    axes[0, 0].set_title("Q-value Evolution (Player 0)")
    axes[0, 0].set_xlabel("Rescaled Time" if rescale else "Time")
    axes[0, 0].set_ylabel("Q-value")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Plot Q-values for Player 1
    for state_id in range(num_states):
        for action_id in range(num_actions):
            # Solution
            q_values_solution = [
                get_q_value(x_solution_temp[1][t], state_id, action_id)
                for t in range(len(t_eval))
            ]
            axes[0, 1].plot(
                x_eval, q_values_solution,
                linestyle=action_styles[action_id],
                color=state_styles[state_id % len(state_styles)],
                label=rf"$Q^1_{{{states_names[state_id]}, {actions_names[action_id]}}}$"
            )
            # Logs
            q_values_logs = [
                get_q_value(q, state_id, action_id)
                for q in logs_mean_Q_temp[1]
            ]
            axes[0, 1].plot(
                x_logs, q_values_logs,
                linestyle=action_styles[action_id],
                color=state_styles[state_id % len(state_styles)],
                alpha=0.4, linewidth=3
            )
    axes[0, 1].set_title("Q-value Evolution (Player 1)")
    axes[0, 1].set_xlabel("Rescaled Time" if rescale else "Time")
    axes[0, 1].set_ylabel("Q-value")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Plot probabilities
    for player_id, reinforcer in enumerate(reinforcers):
        for state_id in range(num_states):
            # Solution
            probabilities_solution = []
            for t in range(len(t_eval)):
                probabilities_solution.append(
                    SMEpolicy(x_solution_temp[player_id][t], state_id, reinforcer.tau, reinforcer.epsilon)[0]
                )
            axes[1, 0].plot(
                x_eval, probabilities_solution,
                color=state_styles[state_id % len(state_styles)],
                linestyle=player_styles[player_id],
                label=rf"$Q^{player_id}_{{{states_names[state_id]}, C}}$"
            )
            # Logs
            probabilities_logs = []
            for q in logs_mean_Q_temp[player_id]:
                probabilities_logs.append(
                    SMEpolicy(q, state_id, reinforcer.tau, reinforcer.epsilon)[0]
                )
            axes[1, 0].plot(
                x_logs, probabilities_logs,
                color=state_styles[state_id % len(state_styles)],
                linestyle=player_styles[player_id],
                alpha=0.4, linewidth=3
            )
    axes[1, 0].set_title("Probability of Cooperation")
    axes[1, 0].set_xlabel("Rescaled Time" if rescale else "Time")
    axes[1, 0].set_ylabel("Probability")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    windows = math.floor(len(logs["state_proportions"][:, 0]) / 1000)
    if windows < 1:
        windows = 1
    elif windows > len(logs["state_proportions"][:, 0]):
        windows = len(logs["state_proportions"][:, 0])
    # Plot state distribution
    for state_id in range(num_states):
        # Solution
        axes[1, 1].plot(
            x_eval, S_solution[state_id, :],
            color=state_styles[state_id % len(state_styles)],
            label=rf"$P({states_names[state_id]})$"
        )
        # Logs
        state_proportions_logs = moving_average(logs["state_proportions"][:, state_id], windows)
        axes[1, 1].plot(
            x_logs_state[:len(state_proportions_logs)], state_proportions_logs,
            color=state_styles[state_id % len(state_styles)],
            alpha=0.4
        )
    axes[1, 1].set_title("State Distribution")
    axes[1, 1].set_xlabel("Rescaled Time" if rescale else "Time")
    axes[1, 1].set_ylabel(f"State Probability - MA of {windows}")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    # Adjust layout
    plt.tight_layout()
    plt.show()