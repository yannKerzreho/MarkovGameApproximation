import numpy as np
from typing import Union, Dict, Tuple, List

class Q:
    """
    A class to handle Q-value operations uniformly for both numpy arrays and dictionaries.
    
    Attributes:
        data: The Q-value data structure (numpy array or dictionary of numpy arrays)
    """
    def __init__(self, data: Union[np.ndarray, Dict[str, np.ndarray]]):
        """
        Initialize Q structure.
        
        :param data: Either a numpy array or a dictionary of numpy arrays
        """
        if not (isinstance(data, np.ndarray) or 
                (isinstance(data, dict) and all(isinstance(v, np.ndarray) for v in data.values()))):
            raise ValueError("Data must be either numpy array or dictionary of numpy arrays")
        
        self.data = data
        self._metadata = self._create_metadata()
    
    def _create_metadata(self) -> dict:
        """
        Create metadata about the Q structure.
        
        :return: Dictionary containing metadata about the structure
        """
        if isinstance(self.data, np.ndarray):
            return {
                'type': 'array',
                'shape': self.data.shape,
                'size': self.data.size
            }
        else:
            shapes = {key: value.shape for key, value in self.data.items()}
            sizes = {key: value.size for key, value in self.data.items()}
            return {
                'type': 'dict',
                'shapes': shapes,
                'sizes': sizes,
                'keys': list(self.data.keys()),
                'total_size': sum(sizes.values())
            }
    
    def copy(self) -> 'Q':
        """
        Create a deep copy of the Q structure.
        
        :return: New Q instance with copied data
        """
        if isinstance(self.data, np.ndarray):
            return Q(self.data.copy())
        else:
            return Q({key: value.copy() for key, value in self.data.items()})
    
    def fill(self, value: float) -> None:
        """
        Fill the Q structure with a value.
        
        :param value: Value to fill with
        """
        if isinstance(self.data, np.ndarray):
            self.data.fill(value)
        else:
            for arr in self.data.values():
                arr.fill(value)
    
    def __add__(self, other: Union['Q', float, int]) -> 'Q':
        """
        Add two Q structures or add a scalar to Q structure.
        
        :param other: Another Q instance or a scalar
        :return: New Q instance with results
        """
        if isinstance(other, (float, int)):
            if isinstance(self.data, np.ndarray):
                return Q(self.data + other)
            return Q({key: value + other for key, value in self.data.items()})
        
        if not isinstance(other, Q):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        
        if isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            return Q(self.data + other.data)
        elif isinstance(self.data, dict) and isinstance(other.data, dict):
            return Q({key: self.data[key] + other.data[key] for key in self.data})
        else:
            raise ValueError("Cannot add Q structures of different types")
    
    def __sub__(self, other: Union['Q', float, int]) -> 'Q':
        """
        Subtract two Q structures or subtract a scalar from Q structure.
        
        :param other: Another Q instance or a scalar
        :return: New Q instance with results
        """
        if isinstance(other, (float, int)):
            if isinstance(self.data, np.ndarray):
                return Q(self.data - other)
            return Q({key: value - other for key, value in self.data.items()})
        
        if not isinstance(other, Q):
            raise TypeError(f"Unsupported operand type: {type(other)}")
        
        if isinstance(self.data, np.ndarray) and isinstance(other.data, np.ndarray):
            return Q(self.data - other.data)
        elif isinstance(self.data, dict) and isinstance(other.data, dict):
            return Q({key: self.data[key] - other.data[key] for key in self.data})
        else:
            raise ValueError("Cannot subtract Q structures of different types")
    
    def __mul__(self, scalar: Union[float, int]) -> 'Q':
        """
        Multiply Q structure by a scalar.
        
        :param scalar: Number to multiply by
        :return: New Q instance with results
        """
        if not isinstance(scalar, (float, int)):
            raise TypeError(f"Can only multiply by scalar values, not {type(scalar)}")
        
        if isinstance(self.data, np.ndarray):
            return Q(self.data * scalar)
        return Q({key: value * scalar for key, value in self.data.items()})
    
    def __truediv__(self, scalar: Union[float, int]) -> 'Q':
        """
        Divide Q structure by a scalar.
        
        :param scalar: Number to divide by
        :return: New Q instance with results
        """
        if not isinstance(scalar, (float, int)):
            raise TypeError(f"Can only divide by scalar values, not {type(scalar)}")
        if scalar == 0:
            raise ValueError("Cannot divide by zero")
        
        if isinstance(self.data, np.ndarray):
            return Q(self.data / scalar)
        return Q({key: value / scalar for key, value in self.data.items()})
    
    def __rmul__(self, scalar: Union[float, int]) -> 'Q':
        """
        Right multiplication by scalar.
        
        :param scalar: Number to multiply by
        :return: New Q instance with results
        """
        return self.__mul__(scalar)
    
    def flatten(self) -> Tuple[np.ndarray, dict]:
        """
        Flatten Q structure into 1D array.
        
        :return: Tuple of (flattened array, metadata for reshaping)
        """
        if isinstance(self.data, np.ndarray):
            return self.data.ravel(), self._metadata
        else:
            flattened = []
            for key in self._metadata['keys']:
                flattened.append(self.data[key].ravel())
            return np.concatenate(flattened), self._metadata
    
    @staticmethod
    def reshape(flat_array: np.ndarray, metadata: dict) -> 'Q':
        """
        Reshape flat array back to Q structure.
        
        :param flat_array: 1D numpy array
        :param metadata: Metadata for reshaping
        :return: New Q instance with reshaped data
        """
        if metadata['type'] == 'array':
            return Q(flat_array.reshape(metadata['shape']))
        else:
            Q_dict = {}
            start_idx = 0
            for key in metadata['keys']:
                shape = metadata['shapes'][key]
                size = metadata['sizes'][key]
                Q_dict[key] = flat_array[start_idx:start_idx + size].reshape(shape)
                start_idx += size
            return Q(Q_dict)
    
    def __repr__(self) -> str:
        """String representation of Q structure."""
        return f"Q({self.data})"

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
    def __init__(self, action_space_size, state_space_size, Q_data):
        """
        Initialize the Reinforcer.

        :param action_space_size: Number of possible actions the agent can take.
        :param state_space_size: Size of the state space.
        :param Q_data: Initial Q values as either numpy array or dictionary of numpy arrays.
        """
        self.action_space_size = action_space_size
        self.state_space_size = state_space_size
        
        # Wrap Q_data in Q class if it isn't already
        self.Q = Q_data if isinstance(Q_data, Q) else Q(Q_data)

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

    def get_Q_copy(self):
        """
        Get a deep copy of the Q structure.

        :return: Copy of Q structure wrapped in Q class
        """
        return self.Q.copy()