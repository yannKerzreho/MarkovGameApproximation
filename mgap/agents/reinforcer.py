import numpy as np
from typing import Union, Dict, Tuple, List

class Reinforcer:
    """
    Represents a generic agent in a Markov game.
    Supports both numpy array and dictionary Q-value structures through the Q class.

    Attributes:
        action_space_size (int): The number of possible actions the agent can take.
        state_space_size (int): The size of the state space.
        policy: Function that determines action probabilities given Q-values and state.
        Q (Q): Q-value structure wrapped in Q class.
    """
    def __init__(self, action_space_size, state_space_size, param):
        """
        Initialize the Reinforcer.

        :param action_space_size: Number of possible actions the agent can take.
        :param state_space_size: Size of the state space.
        :param Q_data: Initial Q values as either numpy array or dictionary of numpy arrays.
        """
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        self.param = param

    def policy(self, current_state):
        """
        Compute the probabilities to play each actions on a given state.

        Args:
            current_state (int): The current state of the agent.

        Returns:
            list: Probabilities list.
        """
        raise NotImplementedError("update must be implemented in a subclass.")
    
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


    def update(self, indix, joint_actions, rewards, current_state, next_state):
        """
        Abstract method to update internal state (e.g., Q) based on experience.
        To be implemented in inherited classes.

        Note: When implementing in subclasses, make sure to wrap any Q-value
        updates using the Q class operations.

        :param indix: Order in the list of player.
        :param joint_actions: Actions taken by players in order.
        :param rewards: Rewards received by players in order.
        :param current_state: Current state of the game.
        :param next_state: Next state of the game.
        """
        raise NotImplementedError("update must be implemented in a subclass.")