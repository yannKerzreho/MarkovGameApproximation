import numpy as np
from Reinforcer import Reinforcer, Q
from scipy.special import logsumexp

class QTableReinforcer(Reinforcer):
    """
    A reinforcer implementing Q-learning with a table-based Q representation.

    Attributes:
        Q (Q): The Q-table, storing the value of each state-action pair.
        alpha (float): Learning rate, determines how new information overrides old information.
        gamma (float): Discount factor, represents the importance of future rewards.
        tau (float): Temperature of the soft-max part of the policy.
        epsilon (float): Probability to chose a random action in the policy.
        policy (function): A policy function mapping Q-values and the current state to action probabilities.
    """

    def __init__(self, action_space_size: int, state_space_size: int, alpha: float, gamma: float, tau: float, epsilon: float, initial_Q=None):
        """
        Initializes the QTableReinforcer with required parameters.

        Args:
            action_space_size (int): Number of possible actions in the environment.
            state_space_size (int): Size of the state space.
            policy (function): Policy function for selecting actions, e.g., epsilon-greedy or softmax.
            alpha (float): Learning rate for the Q-learning algorithm.
            gamma (float): Discount factor for future rewards.
            tau (float): Temperature of the soft-max part of the policy.
            epsilon (float): Probability to chose a random action in the policy.
            initial_Q (numpy.ndarray or Q, optional): Initial Q-values. Defaults to zero if None.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon

        # Handle initial Q-values
        if initial_Q is None:
            initial_Q = Q(np.zeros((state_space_size, action_space_size), dtype=np.float64))
        elif not isinstance(initial_Q, Q):
            initial_Q = Q(initial_Q)
        
        # Ensure Q-values are of float type
        if isinstance(initial_Q.data, np.ndarray):
            assert np.issubdtype(initial_Q.data.dtype, np.floating), "Q-table must contain float values"
        elif isinstance(initial_Q.data, dict):
            assert all(np.issubdtype(v.dtype, np.floating) for v in initial_Q.data.values()), \
                "All Q-table values must be floats"

        super().__init__(action_space_size, state_space_size, initial_Q)

    def policy(self, current_state):
        """
        Compute the probabilities to play each actions on a given state.

        Args:
            current_state (int): The current state of the agent.

        Returns:
            list: Probabilities list.
        """
        state_Q_values = self.Q.data[current_state]
        Q_scaled = state_Q_values / self.tau
        softmax = np.exp(Q_scaled - logsumexp(Q_scaled))
        return softmax * (1 - self.epsilon) + self.epsilon / len(softmax)  # Epsilon-greedy adjustment

    def choose_action(self, current_state: int):
        """
        Selects an action based on the current state and the policy.

        Args:
            current_state (int): The current state of the agent.

        Returns:
            int: The selected action.
        """
        probabilities = self.policy(current_state)
        return np.random.choice(range(self.action_space_size), p=probabilities)

    def update(self, indix: int, joint_actions: list, rewards: list, current_state: int, next_state: int):
        """
        Updates the Q-table based on the agent's experience.

        Args:
            indix (int): Index of the agent in the joint actions list.
            joint_actions (list): List of actions taken by all agents.
            rewards (list): Rewards received by all agents.
            current_state (int): The current state of the environment.
            next_state (int): The next state of the environment.
        """
        action = joint_actions[indix]
        td_error = rewards[indix] + self.gamma * np.max(self.Q.data[next_state]) - self.Q.data[current_state, action]
        self.Q.data[current_state, action] += self.alpha * td_error


class QTableCounterFactualReinforcer(Reinforcer):
    """
    A Q-learning reinforcer with counterfactual updates for multi-agent environments.

    Attributes:
        Q (Q): The Q-table, storing state-action pair values.
        alpha (float): Learning rate.
        gamma (float): Discount factor.
        tau (float): Temperature of the soft-max part of the policy.
        epsilon (float): Probability to chose a random action in the policy.
        policy (function): Policy function for action selection.
        rewards_matrix (dict): A matrix storing joint rewards for state and joint actions.
    """

    def __init__(self, action_space_size: int, state_space_size: int, alpha: float, gamma: float, tau: float, epsilon: float, rewards_matrix, transition_matrix, initial_Q=None):
        """
        Initializes the counterfactual QTableReinforcer.

        Args:
            action_space_size (int): Number of actions available to the agent.
            state_space_size (int): Number of states in the environment.
            policy (function): A policy function mapping Q-values to action probabilities.
            alpha (float): Learning rate.
            gamma (float): Discount factor.
            tau (float): Temperature of the soft-max part of the policy.
            epsilon (float): Probability to chose a random action in the policy.
            rewards_matrix (dict): A matrix storing rewards for each state and joint action combination.
            initial_Q (numpy.ndarray or Q, optional): Initial Q-values. Defaults to zero if None.
        """
        self.alpha = alpha
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.rewards_matrix = rewards_matrix
        self.transition_matrix = transition_matrix

        # Handle initial Q-values
        if initial_Q is None:
            initial_Q = Q(np.zeros((state_space_size, action_space_size), dtype=np.float64))
        elif not isinstance(initial_Q, Q):
            initial_Q = Q(initial_Q)

        # Ensure Q-values are of float type
        if isinstance(initial_Q.data, np.ndarray):
            assert np.issubdtype(initial_Q.data.dtype, np.floating), "Q-table must contain float values"
        elif isinstance(initial_Q.data, dict):
            assert all(np.issubdtype(v.dtype, np.floating) for v in initial_Q.data.values()), \
                "All Q-table values must be floats"

        super().__init__(action_space_size, state_space_size, initial_Q)

    def policy(self, current_state):
        """
        Compute the probabilities to play each actions on a given state.

        Args:
            current_state (int): The current state of the agent.

        Returns:
            list: Probabilities list.
        """
        state_Q_values = self.Q.data[current_state]
        exp_Q = np.exp(state_Q_values / self.tau)  # Compute exponentiated Q-values
        soft_max_probabilities = exp_Q / exp_Q.sum()  # Normalize to create probabilities
        soft_max_probabilities = np.nan_to_num(soft_max_probabilities, nan=1, posinf=1e-10, neginf=1e-10)
        soft_max_probabilities /= soft_max_probabilities.sum()
        return soft_max_probabilities * (1 - self.epsilon) + self.epsilon / len(soft_max_probabilities)  # Epsilon-greedy adjustment

    def choose_action(self, current_state: int):
        """
        Selects an action based on the current state and policy.

        Args:
            current_state (int): The current state of the agent.

        Returns:
            int: The selected action.
        """
        probabilities = self.policy(current_state)
        return np.random.choice(range(self.action_space_size), p=probabilities)
    
    def compute_nextmaxQ(self, joint_actions, current_state):

        probabilities = self.transition_matrix[current_state][tuple(joint_actions)]
        nextmaxQ = 0
        for next_state, prob in enumerate(probabilities):
            nextmaxQ += prob * np.max(self.Q.data[next_state])
        
        return nextmaxQ
        

    def update(self, indix, joint_actions, rewards, current_state, next_state):
        """
        Updates the Q-table using counterfactual reasoning.

        Args:
            indix (int): Index of the agent in the joint actions list.
            joint_actions (list): List of actions taken by all agents.
            rewards (list): Rewards received by all agents.
            current_state (int): The current state of the environment.
            next_state (int): The next state of the environment.
        """
        joint_actions = list(joint_actions)
        for action in range(self.action_space_size):
            joint_actions[indix] = action
            nextmaxQ = self.compute_nextmaxQ(joint_actions, current_state)
            td_error = (
                self.rewards_matrix[current_state][tuple(joint_actions)][indix] 
                + self.gamma * nextmaxQ
                - self.Q.data[current_state, action]
            )
            self.Q.data[current_state, action] += self.alpha * td_error