import poke_env
from poke_env.player import Player
from copy import deepcopy

class Data:
    """
    A class for storing battle state, move data, and associated reward for reinforcement learning.
    """

    def __init__(self, battle: poke_env.environment.battle.Battle, move):
        """
        Initialize a data sample with battle state and selected move.
        
        Args:
            battle: The current battle state to store
            move: The move that was sampled/chosen for this battle state
        """
        self.internal_battle = deepcopy(battle)  # Store a deep copy to prevent modification of the original battle
        self.sampled_move = move
        self.reward = None  # Reward is initially unset
    
    def set_reward(self, reward):
        """
        Set the reward value for this battle state and move pair.
        
        Args:
            reward: The reward value to associate with this state-action pair
        """
        self.reward = reward
    
    # TODO: Provide a function to convert state to pytorch tensor for neural network inferencing/input