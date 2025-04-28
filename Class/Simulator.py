import numpy as np
import copy
from scipy.integrate import solve_ivp
from typing import Union, Dict, Tuple, List
from Class.MarkovGame import MarkovGame
from Class.Reinforcer import Reinforcer, Q

class Simulator:
    """
    Simulates multiple independent runs of a Markov game, logging results and aggregating statistics.
    Provides robust handling of Q-values through the Q class interface.
    
    Attributes:
        all_logs (list): Individual logs from each simulation run
        final_log (dict): Aggregated statistics across all simulation runs, including:
            - mean_Q: Average Q-values/parameters for each iteration
            - mean_rewards: Average rewards for each iteration
            - state_proportions: Average state distributions for each iteration
    """
    def __init__(self):
        """Initialize the Simulator with empty logs."""
        self.all_logs = []
        self.final_log = {
            "mean_Q": [],  # List of Q instances per iteration
            "mean_rewards": [],  # Mean rewards array
            "state_proportions": []  # State distribution array
        }

    def initialize_mean_Q(self, template_Q: Q, num_iterations: int) -> np.ndarray:
        """
        Initialize storage for mean Q values using Q class.
        
        :param template_Q: Q instance to use as template
        :param num_iterations: Number of iterations to store
        :return: Array of Q instances initialized with zeros
        """
        # Get flattened representation and metadata
        flat_template, metadata = template_Q.flatten()
        
        # Create zero-initialized array with additional dimension for iterations
        flat_storage = np.zeros((num_iterations, len(flat_template)))
        
        # Create an array of Q instances, one for each iteration
        mean_Q_array = np.array([
            Q.reshape(flat_storage[i], metadata.copy())
            for i in range(num_iterations)
        ])
        
        return mean_Q_array

    def accumulate_Q(self, mean_Q: Q, new_Q: Q, sim_count: int) -> Q:
        """
        Update running average of Q values using Q class operations.
        
        :param mean_Q: Current mean Q values
        :param new_Q: New Q values to incorporate
        :param sim_count: Number of simulations completed
        :return: Updated mean Q values
        """
        # Use Q class operations for running average
        return (mean_Q * sim_count + new_Q) / (sim_count + 1)

    def run_simulations(self, game: MarkovGame, reinforcers: list, 
                       num_simulations: int, num_iterations: int) -> None:
        """
        Run multiple independent simulations of a Markov game with reinforcers.
        
        :param game: MarkovGame instance
        :param reinforcers: List of Reinforcer objects
        :param num_simulations: Number of independent simulations to run
        :param num_iterations: Number of iterations per simulation
        :raises ValueError: If number of reinforcers doesn't match game players
        """
        if len(reinforcers) != game.num_players:
            raise ValueError("Number of reinforcers must match number of players")

        # Initialize arrays for storing mean values per iteration
        mean_Q_values = [
            self.initialize_mean_Q(reinforcer.Q, num_iterations)
            for reinforcer in reinforcers
        ]
        mean_rewards = np.zeros((num_iterations, game.num_players))
        mean_state_counts = np.zeros((num_iterations, game.state_space_size))

        for sim_idx in range(num_simulations):
            # Create deep copies of reinforcers for this simulation
            current_reinforcers = [copy.deepcopy(r) for r in reinforcers]
            
            # Reset game state
            game.reset_log()
            game.set_state(np.random.randint(0, game.state_space_size))
            
            # Run simulation
            game.run(num_iterations, current_reinforcers)
            sim_log = game.get_logs()
            self.all_logs.append(sim_log)

            # Update statistics for each iteration
            for iter_idx in range(num_iterations):
                # Update rewards
                mean_rewards[iter_idx] += (
                    sim_log["rewards"][iter_idx] - mean_rewards[iter_idx]
                ) / (sim_idx + 1)

                # Update Q values for each player at each iteration
                for player_idx, reinforcer in enumerate(current_reinforcers):
                    mean_Q_values[player_idx][iter_idx] = self.accumulate_Q(
                        mean_Q_values[player_idx][iter_idx],
                        sim_log["Q_values"][player_idx][iter_idx],
                        sim_idx
                    )

                # Update state proportions
                state_idx = sim_log["states"][iter_idx]
                current_counts = np.zeros(game.state_space_size)
                current_counts[state_idx] = 1
                mean_state_counts[iter_idx] += (
                    current_counts - mean_state_counts[iter_idx]
                ) / (sim_idx + 1)

        # Store final statistics
        self.final_log["mean_Q"] = mean_Q_values
        self.final_log["mean_rewards"] = mean_rewards
        self.final_log["state_proportions"] = mean_state_counts