import poke_env
from poke_env.player import Player
from copy import deepcopy
from PokemonData import Pokemon_Indices, Ability_Indices, Item_Indices, Move_Indices
import torch


class PokemonData:
    """
    A class for storing all relevant data about a specific pokemon.
    Provides functionality to convert pokemon attributes into a tensor representation
    for machine learning models.
    """
    def __init__(self, pokemon=None, is_our_pokemon=False):
        """
        Initialize a pokemon data object either from an existing pokemon object or with default values.
        
        Args:
            pokemon: A pokemon object with all the required attributes. If None, empty pokemon will be created.
            is_our_pokemon: Boolean flag indicating if this pokemon belongs to the player. Defaults to False.
        """
        if pokemon is not None:
            # Initialize with data from an existing pokemon object
            self.name = pokemon.name                                # 420 options
            self.moves = pokemon.moves                              # 4 of 351 options
            self.item = pokemon.item                                # 63 options
            self.ability = pokemon.ability                          # 209 options
            self.active = pokemon.active                            # Boolean
            self.boosts = pokemon.boosts                            # 7 values
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
            self.is_our_pokemon = is_our_pokemon                    # Boolean
        else:
            # Initialize an empty pokemon data object with default values
            self.name = None
            self.moves = []
            self.item = None
            self.ability = None
            self.boosts = {"accuracy": 0, "atk": 0, "def": 0, "evasion": 0, "spa": 0, "spd": 0, "spe": 0}  # Fixed typo in "accuracy"
            self.active = False  # Fixed from the original which had boosts data in active
            self.hp = 1.0
            self.effects = {}
            self.fainted = False
            self.is_terastallized = False
            self.tera_type = None
            self.protect_counter = 0
            self.revealed = False
            self.status = None
            self.status_counter = 0
            self.types = None
            self.is_our_pokemon = is_our_pokemon
    
    def return_tensor(self):
        """
        Convert pokemon attributes into a one-hot encoded tensor representation.
        
        Returns:
            torch.Tensor: A binary vector representing the pokemon's attributes
                          for use in machine learning models
        """
        tensor = torch.zeros([
            len(Pokemon_Indices.keys()) + 
            len(Move_Indices.keys()) + 
            len(Item_Indices.keys()) + 
            len(Ability_Indices.keys()) + 
            1 +                                                 # Is active
            1 +                                                 # HP
            1 +                                                 # Fainted
            1 +                                                 # Revealed
            1                                                   # Is our pokemon
        ])

        # One-hot encode the pokemon name
        tensor[Pokemon_Indices[self.name]] = 1
        head = len(Pokemon_Indices.keys())

        # One-hot encode each move the pokemon has
        for move in self.moves:
            tensor[Move_Indices[move] + head] = 1
        head += len(Move_Indices.keys())

        # One-hot encode the held item
        tensor[Item_Indices[self.item] + head] = 1
        head += len(Item_Indices.keys())

        # One-hot encode the ability
        tensor[Ability_Indices[self.ability] + head] = 1
        head += len(Ability_Indices.keys())

        # Binary features
        if self.active:
            tensor[head] = 1
        head += 1

        # Continuous HP value
        tensor[head] = self.hp
        head += 1

        # More binary features
        if self.fainted:
            tensor[head] = 1
        head += 1

        if self.revealed:
            tensor[head] = 1
        head += 1

        if self.is_our_pokemon:
            tensor[head] = 1
        head += 1

        return tensor
        


class DataPoint:
    """
    A class for storing battle state, move data, and associated reward.
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