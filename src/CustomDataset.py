import poke_env
from poke_env.player import Player
from copy import deepcopy


class PokemonData:
    def __init__(self, pokemon, is_our_pokemon):
        self.name = pokemon.name                                # 420 options
        self.moves = pokemon.moves                              # 4 of 351 options
        self.item = pokemon.item                                # 62 options
        self.ability = pokemon.ability                          # 208 options
        self.active = pokemon.active                            # Boolean
        self.boosts = pokemon.boosts                            # 8 values
        self.hp = pokemon.current_hp_fraction                   # 1 float in [0, 1]
        self.effects = pokemon.effects                          # 224 options
        self.fainted = pokemon.fainted                          # Boolean
        self.is_terastallized = pokemon.is_terastallized        # Boolean
        self.tera_type = pokemon.tera_type                      # One of 20
        self.protect_counter = pokemon.protect_counter          # Int
        self.revealed = pokemon.revealed                        # Boolean
        self.status = pokemon.status                            # 1 of 7
        self.status_counter = pokemon.status_counter            # Int
        self.types = pokemon.types                              # 1-3 of 20
        self.is_our_pokemon = is_our_pokemon


class DataPoint:
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
        self.our_team = [
            PokemonData(pokemon, True) for pokemon in battle.team.values()
        ]
        self.opp_team = [
            PokemonData(pokemon, False) for pokemon in battle.opponent_team.values()
        ]

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