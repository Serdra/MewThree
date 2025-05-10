import poke_env
import random
from PokemonData import Pokemon_Indices, Ability_Indices, Item_Indices, Move_Indices, Num_Pokemon, Num_Abilities, Num_Items, Num_Moves
import torch
from typing import List, Tuple


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
    
    @staticmethod
    def get_tensor_length():
        return (len(Pokemon_Indices.keys()) + 
            len(Move_Indices.keys()) + 
            len(Item_Indices.keys()) + 
            len(Ability_Indices.keys()) + 
            1 +                                                 # Is active
            1 +                                                 # HP
            1 +                                                 # Fainted
            1 +                                                 # Revealed
            1)                                                  # Is our pokemon
        

class DataPoint:
    """
    A class for storing battle state, move data, and associated reward.
    """

    def __init__(self, battle: poke_env.environment.battle.Battle):
        """
        Initialize a data sample with battle state and selected move.
        
        Args:
            battle: The current battle state to store
        """
        self.our_team = [
            PokemonData(pokemon, True) for pokemon in battle.team.values()
        ]

        self.opp_team = [
            PokemonData(pokemon, False) for pokemon in battle.opponent_team.values()
        ]

        while len(self.our_team) < 6:
            self.our_team.append(PokemonData(None, True))
        while len(self.opp_team) < 6:
            self.opp_team.append(PokemonData(None, False))

        self.sampled_move = None
        self.reward = None  # Reward is initially unset
    
    def set_move(self, move: int):
        """
        Set the move made during the stored state.
        
        Args:
            move: The move made
        """
        self.sampled_move = move
    
    def set_reward(self, reward: float):
        """
        Set the reward value for this battle state and move pair.
        
        Args:
            reward: The reward value to associate with this state-action pair
        """
        self.reward = reward

    def get_input(self):
        """
        Gets the input to pass to a neural network for the position. Indices 0 and 6 are reserved for active pokemon
        if applicable. Indices 0-5 are our pokemon and 6-11 are the opponents. The order is randomized between all
        non-active pokemon.
        """
        tensor = torch.empty(12, PokemonData.get_tensor_length())
        
        # Create copies of our team and opponent team for manipulation
        our_team_copy = self.our_team.copy()
        opp_team_copy = self.opp_team.copy()
        
        # Find and remove active Pokémon from our team
        active_index_our = None
        for i, pokemon in enumerate(our_team_copy):
            if pokemon.active:
                active_index_our = i
                break
        
        # If we found an active Pokémon, put it at index 0
        if active_index_our is not None:
            active_pokemon = our_team_copy.pop(active_index_our)
            tensor[0] = active_pokemon.return_tensor()
        else:
            # If no active Pokémon (shouldn't happen in normal gameplay), use first Pokémon
            tensor[0] = our_team_copy.pop(0).return_tensor()
        
        # Find and remove active Pokémon from opponent team
        active_index_opp = None
        for i, pokemon in enumerate(opp_team_copy):
            if pokemon.active:
                active_index_opp = i
                break
        
        # If we found an active Pokémon, put it at index 6
        if active_index_opp is not None:
            active_pokemon = opp_team_copy.pop(active_index_opp)
            tensor[6] = active_pokemon.return_tensor()
        else:
            # If no active Pokémon (shouldn't happen in normal gameplay), use first Pokémon
            tensor[6] = opp_team_copy.pop(0).return_tensor()
        
        # Randomize the order of remaining Pokémon in our team
        random.shuffle(our_team_copy)
        # Fill indexes 1-5 with our remaining Pokémon
        for i in range(1, 6):
            if i-1 < len(our_team_copy):
                tensor[i] = our_team_copy[i-1].return_tensor()
            else:
                # This shouldn't happen if we properly pad the teams to 6
                tensor[i] = PokemonData(None, True).return_tensor()
        
        # Randomize the order of remaining Pokémon in opponent team
        random.shuffle(opp_team_copy)
        # Fill indexes 7-11 with opponent's remaining Pokémon
        for i in range(7, 12):
            if i-7 < len(opp_team_copy):
                tensor[i] = opp_team_copy[i-7].return_tensor()
            else:
                # This shouldn't happen if we properly pad the teams to 6
                tensor[i] = PokemonData(None, False).return_tensor()
        
        return tensor


class CustomDataset:
    """
    A fixed-size, ring-buffer dataset for storing and sampling data points.
    
    Used for experience replay in reinforcement learning, where older experiences
    are overwritten when the buffer reaches capacity.
    """
    def __init__(self, size=50000):
        """
        Initialize an empty dataset with specified maximum size.
        
        Args:
            size: Maximum number of samples to store in the dataset (default: 50000)
        """
        self.size = size
        self.data = []            # List to store data points
        self.head = 0             # Current position in the ring buffer
        self.samples_since_last_step = 0  # Counter for tracking new samples
    
    def add_sample(self, sample: DataPoint) -> None:
        """
        Add a single data point to the dataset.
        
        If the dataset is not full, append the sample.
        If the dataset is full, replace the oldest sample.
        
        Args:
            sample: A DataPoint object to add to the dataset
        """
        if len(self.data) < self.size:
            self.data.append(sample)
        else:
            self.data[self.head] = sample
        
        self.head += 1
        self.head %= self.size  # Wrap around when reaching the end of the buffer
        
        self.samples_since_last_step += 1
    
    def add_samples(self, samples: List) -> None:
        """
        Add multiple data points to the dataset.
        
        Args:
            samples: A list of DataPoint objects to add to the dataset
        """
        for sample in samples:
            self.add_sample(sample)
    
    def get_sample(self) -> Tuple:
        """
        Randomly sample a data point from the dataset.
        
        Returns:
            tuple: A tuple containing (input_data, sampled_move, reward)
        """
        sample = random.choice(self.data)
        return sample.get_input(), sample.sampled_move, sample.reward