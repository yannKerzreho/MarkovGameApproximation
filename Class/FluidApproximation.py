import numpy as np
import copy
from scipy.integrate import solve_ivp
from typing import Union, Dict, Tuple, List
from Class.MarkovGame import MarkovGame
from Class.Reinforcer import Reinforcer, Q

class FluidApproximation:
    """
    Represents the fluid approximation for a Markov game with reinforcers.
    Uses Q class for Q-value operations consistently.

    Attributes:
        markov_game (MarkovGame): The Markov game being approximated
        reinforcers (list): List of Reinforcer objects
        state_space_size (int): Size of the state space
        action_space_sizes (list): List of action space sizes for each player
        num_players (int): Number of players in the game
        T (numpy.ndarray): Transition matrix from the Markov game
        R (numpy.ndarray): Reward matrix from the Markov game
    """
    def __init__(self, markov_game: MarkovGame, reinforcers: list):
        """
        Initialize the fluid approximation.

        :param markov_game: MarkovGame instance
        :param reinforcers: List of Reinforcer instances
        :raises TypeError: If inputs are not of correct type
        """
        if not isinstance(markov_game, MarkovGame):
            raise TypeError(f"Expected MarkovGame but got {type(markov_game)}")
        if not all(isinstance(r, Reinforcer) for r in reinforcers):
            raise TypeError("All reinforcers must be instances of Reinforcer class")
        if len(reinforcers) != markov_game.num_players:
            raise ValueError("Number of reinforcers must match number of players in game")

        self.markov_game = markov_game
        self.reinforcers = reinforcers
        self.state_space_size = markov_game.state_space_size
        self.action_space_sizes = [r.action_space_size for r in reinforcers]
        self.num_players = markov_game.num_players
        self.T = markov_game.transition_matrix
        self.R = markov_game.reward_matrix

    def compute_F(self, x: list, S: np.ndarray) -> tuple:
        """
        Compute the fluid approximation function F(x).
        
        :param x: List of Q instances representing current Q-values for each player
        :param S: The state distribution vector as numpy array
        :return: Tuple (expected_D, delta_S) where expected_D is list of Q instances 
                and delta_S is numpy array
        :raises ValueError: If input dimensions are incorrect
        """
        if len(S) != self.state_space_size:
            raise ValueError("S must have length equal to the state space size")
        if len(x) != self.num_players:
            raise ValueError("Length of x must match number of players")
        if not all(isinstance(Q_struct, Q) for Q_struct in x):
            raise TypeError("All elements in x must be Q instances")
        
        # Initialize containers using Q class copy operation
        expected_D = [Q_struct.copy() for Q_struct in x]
        for D in expected_D:
            D.fill(0)
        
        delta_S = np.zeros_like(S)

        for i, reinforcer in enumerate(self.reinforcers):
            reinforcer.Q = x[i].copy()
        
        # Compute for each state
        for state in range(self.state_space_size):
            # Get policies for current state

            policies = [
                reinforcer.policy(state)
                for i, reinforcer in enumerate(self.reinforcers)
            ]
            
            # Compute joint policy probabilities
            for joint_actions in np.ndindex(*self.action_space_sizes):
                # Calculate probability of joint action
                action_prob = np.prod([
                    policies[i][joint_actions[i]]
                    for i in range(self.num_players)
                ])
                
                state_prob = S[state] * action_prob
                
                if state_prob > 0:  # Skip computation if probability is zero
                    # Get rewards and compute D values
                    rewards = self.R[state][joint_actions]
                    D_values = self.compute_D(joint_actions, rewards, x, state)
                    
                    # Update expected_D using Q class operations
                    for i in range(len(expected_D)):
                        expected_D[i] += state_prob * D_values[i]

                    # Update delta_S
                    delta_S += state_prob * self.T[state][joint_actions]

        delta_S -= S
        return expected_D, delta_S

    def compute_D(self, joint_actions: tuple, rewards: np.ndarray, x: list, 
                 current_state: int) -> list:
        """
        Compute D values for given state and joint action.
        
        :param joint_actions: Tuple of joint actions
        :param rewards: Rewards received by each player
        :param x: List of Q instances for each player
        :param current_state: Current state index
        :return: List of D values as Q instances
        """
        D = []
        probabilities = self.T[current_state][joint_actions]

        for i, reinforcer in enumerate(self.reinforcers):
            # Initialize D_i using Q class
            D_i = x[i].copy()
            D_i.fill(0)
            
            # Compute D values for each possible next state
            for next_state, prob in enumerate(probabilities):
                if prob > 0:
                    # Create temporary reinforcer with copied Q values
                    temp_reinforcer = copy.deepcopy(reinforcer)
                    temp_reinforcer.update(i, joint_actions, rewards, current_state, next_state)
                    # Use Q class operations for difference
                    D_i += prob * (temp_reinforcer.Q - x[i])
            
            D.append(D_i)
                    
        return D

    def solve_differential_system_naive(self, x0: list, S0: np.ndarray, 
                                t_span: tuple, t_eval: np.ndarray) -> tuple:
        """
        Solves the differential system using solve_ivp.
        
        :param x0: Initial Q-values as list of Q instances
        :param S0: Initial state distribution as numpy array
        :param t_span: Time span for integration as (t_start, t_end)
        :param t_eval: Points at which to evaluate the solution
        :return: Tuple (x_solution, S_solution) where x_solution is list of lists of Q instances
                and S_solution is numpy array
        :raises ValueError: If input dimensions are incorrect
        """
        # Validate inputs
        if len(x0) != self.num_players:
            raise ValueError("Length of x0 must match number of players")
        if len(S0) != self.state_space_size:
            raise ValueError("Length of S0 must match state space size")
        
        # Flatten all Q-values and store metadata
        flattened_x0 = []
        metadata_list = []
        
        for Q_struct in x0:
            flat_Q, metadata = Q_struct.flatten()
            flattened_x0.append(flat_Q)
            metadata_list.append(metadata)

        # Combine all flattened arrays
        y0 = np.concatenate(flattened_x0 + [S0])

        def system(t, y):
            """Define the differential system."""
            current_idx = 0
            current_x = []
            
            # Reconstruct x structures using Q class
            for metadata in metadata_list:
                size = metadata['total_size'] if metadata['type'] == 'dict' else metadata['size']
                flat_piece = y[current_idx:current_idx + size]
                current_x.append(Q.reshape(flat_piece, metadata))
                current_idx += size

            current_S = y[current_idx:]

            # Compute derivatives
            expected_D, delta_S = self.compute_F(current_x, current_S)
            
            # Flatten derivatives
            dx_dt = []
            for D in expected_D:
                flat_D, _ = D.flatten()
                dx_dt.append(flat_D)

            return np.concatenate(dx_dt + [delta_S])

        # Solve the system
        sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45')

        # Reconstruct solution using Q class
        x_solution = []
        current_idx = 0
        
        for metadata in metadata_list:
            size = metadata['total_size'] if metadata['type'] == 'dict' else metadata['size']
            flat_piece = sol.y[current_idx:current_idx + size, :]
            
            # Reshape for each time point using Q class
            player_solution = []
            for t in range(len(t_eval)):
                player_solution.append(Q.reshape(flat_piece[:, t], metadata))
            x_solution.append(player_solution)
            current_idx += size

        S_solution = sol.y[current_idx:, :]

        return x_solution, S_solution

    
    def solve_differential_system_invariant(self, x0: list, 
                                t_span: tuple, t_eval: np.ndarray) -> tuple:
        """
        Solves the differential system using solve_ivp.
        
        :param x0: Initial Q-values as list of Q instances
        :param t_span: Time span for integration as (t_start, t_end)
        :param t_eval: Points at which to evaluate the solution
        :return: Tuple (x_solution, S_solution) where x_solution is list of lists of Q instances
                and S_solution is numpy array
        :raises ValueError: If input dimensions are incorrect
        """
        # Validate inputs
        if len(x0) != self.num_players:
            raise ValueError("Length of x0 must match number of players")

        # Flatten all Q-values and store metadata
        flattened_x0 = []
        metadata_list = []
        
        for Q_struct in x0:
            flat_Q, metadata = Q_struct.flatten()
            flattened_x0.append(flat_Q)
            metadata_list.append(metadata)

        y0 = np.concatenate(flattened_x0)

        def system(t, y):
            """Define the differential system."""
            current_idx = 0
            current_x = []
            
            # Reconstruct x structures using Q class
            for metadata in metadata_list:
                size = metadata['total_size'] if metadata['type'] == 'dict' else metadata['size']
                flat_piece = y[current_idx:current_idx + size]
                current_x.append(Q.reshape(flat_piece, metadata))
                current_idx += size

            # Compute derivatives
            expected_D = []

            for i, reinforcer in enumerate(self.reinforcers):
                reinforcer.Q = current_x[i].copy()

            probabilities = self.markov_game.invar_prob(self.reinforcers)
            probabilities = probabilities / probabilities.sum()

            for i, reinforcer in enumerate(self.reinforcers):

                D_i = current_x[i].copy()
                D_i.fill(0)
                
                # print(f'Debug : Algo : {i} \n')

                for state in range(self.state_space_size):
                    for joint_actions in np.ndindex(*self.action_space_sizes):
                        for next_state in range(self.state_space_size):
                            rewards = self.R[state][joint_actions]
                            # Create temporary reinforcer with copied Q values
                            temp_reinforcer = copy.deepcopy(reinforcer)
                            temp_reinforcer.update(i, joint_actions, rewards, state, next_state)
                            # Use Q class operations for difference
                            # print(f'Debug : state {state} | joint actions {joint_actions} | next state {next_state} \n Proba = {probabilities[state][joint_actions][next_state]}')
                            D_i += probabilities[state][joint_actions][next_state] * (temp_reinforcer.Q - current_x[i])    
                
                # print(f'Debug : expected_D = {expected_D} \n')

                expected_D.append(D_i)       
            
            # Flatten derivatives
            dx_dt = []
            for D in expected_D:
                flat_D, _ = D.flatten()
                dx_dt.append(flat_D)

            return np.concatenate(dx_dt)

        # Solve the system
        sol = solve_ivp(system, t_span, y0, t_eval=t_eval, method='RK45')

        # Reconstruct solution using Q class
        x_solution = []
        current_idx = 0
        
        for metadata in metadata_list:
            size = metadata['total_size'] if metadata['type'] == 'dict' else metadata['size']
            flat_piece = sol.y[current_idx:current_idx + size, :]
            
            # Reshape for each time point using Q class
            player_solution = []
            for t in range(len(t_eval)):
                player_solution.append(Q.reshape(flat_piece[:, t], metadata))
            x_solution.append(player_solution)
            current_idx += size

        S_solution = []

        for t in range(len(t_eval)):
            for i, reinforcer in enumerate(self.reinforcers):
                reinforcer.Q = x_solution[i][t].copy()

            probabilities = self.markov_game.invar_prob(self.reinforcers)
            state_distrib = probabilities.sum(axis=(1,2,3))
            S_solution.append(state_distrib) 

        return x_solution, np.array(S_solution).T