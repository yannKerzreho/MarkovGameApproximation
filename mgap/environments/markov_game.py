import numpy as np
import copy
from typing import Union, Dict, Tuple, List

class MarkovGame:
    """
    Represents a Markov game with multiple players and state-dependent transitions.

    Attributes:
        state_space_size (int): The number of possible states in the game.
        transition_matrix (numpy.ndarray): A state transition matrix of shape 
            (state_space_size, state_space_size), where probabilities determine the next state.
        reward_matrix (numpy.ndarray): A reward matrix of shape 
            (state_space_size, action_space_size_1, action_space_size_2, ..., num_players).
        num_players (int): The number of players in the game.
        log (dict): A dictionary storing game logs, including actions, rewards, states, and Q-values.
        current_state (int): The current state of the game.
    """
    def __init__(self, state_space_size, transition_matrix, reward_matrix):
        """
        Initialize the Markov game.

        :param state_space_size: The number of possible states in the game.
        :param transition_matrix: A transition matrix of shape 
            (state_space_size, state_space_size) defining the probabilities of state transitions.
        :param reward_matrix: A reward matrix of shape 
            (state_space_size, action_space_size_1, ..., action_space_size_num_players, num_players).
        """
        self.state_space_size = state_space_size
        self.transition_matrix = transition_matrix
        self.reward_matrix = reward_matrix
        self.num_players = len(reward_matrix[0, 0])  # Number of players inferred from matrix shape
        self.log = {
            "actions": [],  # List of joint actions for each round
            "rewards": [],  # List of rewards for each player in each round
            "states": [],  # States for each round
            "Q_values": [[] for _ in range(self.num_players)]  # Q-values for each player at each round
        }
        self.current_state = 0  # Initial state

    def set_state(self, new_state):
        """
        Set the current state of the game.

        :param new_state: The state to set as the current state.
        """
        assert 0 <= new_state < self.state_space_size, "New state must be within the state space."
        self.current_state = new_state

    def play_round(self, reinforcers):
        """
        Simulate a single round of the game.

        :param reinforcers: A list of Reinforcer objects, one for each player.
        :return: A tuple containing joint actions and rewards.
        """
        assert len(reinforcers) == self.num_players, "Number of reinforcers must match the number of players."

        # Each player chooses an action based on their policy
        joint_actions = [reinforcer.choose_action(self.current_state) for reinforcer in reinforcers]

        # Compute rewards for each player based on the current state and joint actions
        rewards = self.reward_matrix[self.current_state][tuple(joint_actions)].copy()

        # Determine the next state based on transition probabilities
        probabilities = self.transition_matrix[self.current_state][tuple(joint_actions)]
        next_state = np.random.choice(range(self.state_space_size), p=probabilities)

        # Update each reinforcer
        for i, reinforcer in enumerate(reinforcers):
            reinforcer.update(i, joint_actions, rewards, self.current_state, next_state)

        # Update the current state of the game
        self.current_state = next_state

        return joint_actions, rewards

    def run(self, iterations, reinforcers):
        """
        Simulate the game for a given number of iterations and log the results.

        :param iterations: Number of iterations to simulate.
        :param reinforcers: A list of Reinforcer objects, one for each player.
        """
        assert len(reinforcers) == self.num_players, "Number of reinforcers must match the number of players."

        for _ in range(iterations):
            # Log the current state and Q-values before the round
            self.log["states"].append(self.current_state)
            for i, reinforcer in enumerate(reinforcers):
                self.log["Q_values"][i].append(reinforcer.param.copy())

            # Play one round
            joint_actions, rewards = self.play_round(reinforcers)

            # Log the actions and rewards
            self.log["actions"].append(joint_actions)
            self.log["rewards"].append(rewards)
    
    def get_logs(self):
        """
        Retrieve the logs of the simulation.

        :return: A dictionary containing the logs.
        """
        return self.log
    
    def reset_log(self):
        """
        Reset the logs of the simulation.

        :return: None
        """
        self.log = {
            "actions": [],
            "rewards": [],
            "states": [],
            "Q_values": [[] for _ in range(self.num_players)]
        }
    
    def invar_prob(self, reinforcers):
        """
        Compute the stationary distribution of (s_n, a_n, s_{n+1}) in a Markov game.
        
        The stationary distribution gives the long-term probability of being in each
        state-action-nextstate triplet (s, a, s') under the current policies.
        
        Parameters:
        -----------
        reinforcers : list
            List of Reinforcer objects.
            
        Returns:
        --------
        invariant_distribution : np.array of shape (|S|, |A_1|, ..., |A_N|, |S|)
            The stationary distribution where:
            - First dimension is the current state
            - Middle dimensions are actions for each player
            - Last dimension is the next state
            Entry [s,a1,...,aN,s'] gives probability of being in state s, 
            taking joint action (a1,...,aN), and transitioning to s'.
        """

        assert len(reinforcers) == self.num_players, "Number of reinforcers must match the number of players."

        num_states = self.transition_matrix.shape[0]
        action_shapes = self.transition_matrix.shape[1:-1]
        num_actions = np.prod(action_shapes)
        
        # Flatten (s, a) -> single index
        num_entries = num_states * num_actions * num_states
        P = np.zeros((num_entries, num_entries))
        
        def index(s, a, s_next):
            "Convert (s, a, s') tuple to a single index"
            a_index = np.ravel_multi_index(a, action_shapes)
            return s * (num_actions * num_states) + a_index * num_states + s_next
        
        # Build the transition matrix P.
        # Given a current state (ps, pa, s),
        # the next state will be (s, a, s') where:
        # - a is chosen according to the players' policies at state s,
        # - s' is drawn from T(s, a).
        for past_state in range(num_states):
            for past_actions in np.ndindex(*action_shapes):
                for state in range(num_states):
                    # Current chain state is (past_state, past_actions, state).
                    # Now, from state, the next joint action is chosen.
                    # Get policies from each reinforcer at state.
                    # Each policy returns a probability distribution over its action space.
                    policies = [reinforcer.policy(state) for reinforcer in reinforcers]

                    # For each possible next joint action a
                    for actions in np.ndindex(*action_shapes):
                        # Joint action probability is the product over players.
                        joint_a_prob = np.prod([policies[i][actions[i]] for i in range(self.num_players)])
                        
                        # For each possible next state s' after taking action a' in state
                        for next_state in range(num_states):
                            # Transition probability from state with actions.
                            trans_prob = self.transition_matrix[(state,) + actions + (next_state,)]
                            
                            # The probability of transitioning from (past_state, past_actions, state) to (state, actions, next_state)
                            p = joint_a_prob * trans_prob
                            
                            # Set the transition probability in the matrix
                            i_from = index(past_state, past_actions, state)
                            i_to = index(state, actions, next_state)
                            P[i_from, i_to] = p

        # Solve for the stationary distribution.
        # We use the eigenvector corresponding to eigenvalue 1 of P.T.
        eigvals, eigvecs = np.linalg.eig(P.T)
        
        # Find the eigenvector whose eigenvalue is (close to) 1.
        stat_vec = np.real(eigvecs[:, np.isclose(eigvals, 1)][:,0])
        stat_vec = stat_vec / np.sum(stat_vec)  # Normalize
        
        # Reshape the stationary distribution into (num_states, *action_shapes, num_states)
        invariant_distribution = stat_vec.reshape((num_states, *action_shapes, num_states))
        return invariant_distribution