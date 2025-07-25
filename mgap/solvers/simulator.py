import numpy as np
import copy
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Union, Dict, Tuple, List
from mgap.environments.markov_game import MarkovGame

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

    def initialize_mean_Q(self, template_Q: np.ndarray, num_iterations: int) -> np.ndarray:
        """
        Initialize storage for mean Q values using numpy arrays.
        
        :param template_Q: numpy array to use as template (shape: state_space x action_space)
        :param num_iterations: Number of iterations to store
        :return: 3D numpy array (num_iterations x state_space x action_space)
        """
        shape = (num_iterations,) + template_Q.shape
        return np.zeros(shape, dtype=np.float64)

    def accumulate_Q(self, mean_Q: np.ndarray, new_Q: np.ndarray, sim_count: int) -> np.ndarray:
        """
        Update running average of Q values using numpy arrays.
        
        :param mean_Q: Current mean Q values (2D array)
        :param new_Q: New Q values to incorporate (2D array)
        :param sim_count: Number of simulations completed
        :return: Updated mean Q values (2D array)
        """
        return (mean_Q * sim_count + new_Q) / (sim_count + 1)
    
    def _simulate_once(self, game_template, reinforcer_templates, num_iterations, seed):
        # Create isolated instances for this process
        game = copy.deepcopy(game_template)
        reinforcers = [copy.deepcopy(r) for r in reinforcer_templates]
        np.random.seed(seed)

        game.reset_log()
        game.set_state(np.random.randint(0, game.state_space_size))
        game.run(num_iterations, reinforcers)
        return game.get_logs()

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

        mean_Q_values = [
            self.initialize_mean_Q(reinforcer.param, num_iterations)
            for reinforcer in reinforcers
        ]
        mean_rewards = np.zeros((num_iterations, game.num_players))
        mean_state_counts = np.zeros((num_iterations, game.state_space_size))

        # Prepare the parallel worker
        seeds = np.random.randint(0, 1e9, size=num_simulations)
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(
                self._simulate_once,
                game,
                reinforcers,
                num_iterations,
                int(seeds[i])
            ) for i in range(num_simulations)]

            for sim_idx, future in enumerate(futures):
                sim_log = future.result()
                self.all_logs.append(sim_log)

                for iter_idx in range(num_iterations):
                    mean_rewards[iter_idx] += (
                        sim_log["rewards"][iter_idx] - mean_rewards[iter_idx]
                    ) / (sim_idx + 1)

                    for player_idx in range(game.num_players):
                        mean_Q_values[player_idx][iter_idx] = self.accumulate_Q(
                            mean_Q_values[player_idx][iter_idx],
                            sim_log["Q_values"][player_idx][iter_idx],
                            sim_idx
                        )

                    state_idx = sim_log["states"][iter_idx]
                    current_counts = np.zeros(game.state_space_size)
                    current_counts[state_idx] = 1
                    mean_state_counts[iter_idx] += (
                        current_counts - mean_state_counts[iter_idx]
                    ) / (sim_idx + 1)

        self.final_log["mean_Q"] = mean_Q_values
        self.final_log["mean_rewards"] = mean_rewards
        self.final_log["state_proportions"] = mean_state_counts