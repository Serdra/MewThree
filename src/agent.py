import poke_env
from poke_env.player import Player
from copy import deepcopy
from CustomDataset import DataPoint
import sys
import random
from PokemonData import Pokemon_Indices, Ability_Indices, Item_Indices, Move_Indices, Num_Pokemon, Num_Abilities, Num_Items, Num_Moves


def sample_with_temperature(data, temperature=1.0):
    """
    Choose randomly from each key and index (0 or 1) based on probability formula.
    
    Args:
        data: Dictionary with keys mapped to tuples of two values in range [0, 1]
        temperature: Controls the randomness of selection (default=1.0)
                     Lower temperature makes selection more deterministic
                     
    Returns:
        Tuple of (selected_key, selected_index)
    """
    # Special case for temperature = 0
    if temperature == 0:
        # Find the key with the highest value in either position
        max_key = None
        max_index = None
        max_value = -1
        
        for key, (val0, val1) in data.items():
            if val0 > max_value:
                max_value = val0
                max_key = key
                max_index = 0
            if val1 > max_value:
                max_value = val1
                max_key = key
                max_index = 1
        
        return max_key, max_index
    
    # Regular case with temperature > 0
    options = []
    weights = []
    
    for key, (val0, val1) in data.items():
        options.append((key, 0))
        options.append((key, 1))
        
        # Apply temperature scaling using the formula value^(1/temperature)
        weight0 = val0 ** (1 / temperature)
        weight1 = val1 ** (1 / temperature)
        
        weights.append(weight0)
        weights.append(weight1)
    
    # Normalize weights to sum to 1
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    # Make random selection
    selected_option = random.choices(options, weights=normalized_weights, k=1)[0]
    return selected_option


class Agent(Player):
    """
    A Pokémon battle agent that extends the poke-env Player class.
    
    This agent can collect data during battles for training purposes.
    """
    def __init__(self, 
                account_configuration = None, 
                *, 
                avatar = None, 
                battle_format = "gen9randombattle", 
                log_level = None, 
                max_concurrent_battles = 1, 
                accept_open_team_sheet = False, 
                save_replays = False, 
                server_configuration = None, 
                start_timer_on_battle_start = False, 
                start_listening = True, 
                open_timeout = 10, 
                ping_interval = 20, 
                ping_timeout = 20, 
                team = None
            ):
        """
        Initialize the Agent with the same parameters as the base Player class.
        
        Args:
            account_configuration: Configuration for the player's account
            avatar: Player avatar ID
            battle_format: Format of battles to play
            log_level: Logging level
            max_concurrent_battles: Maximum number of battles to play simultaneously
            accept_open_team_sheet: Whether to accept open team sheets
            save_replays: Whether to save battle replays
            server_configuration: Server configuration
            start_timer_on_battle_start: Whether to start the timer at battle start
            start_listening: Whether to start listening for battles immediately
            open_timeout: Timeout for opening connections
            ping_interval: Interval between pings
            ping_timeout: Timeout for pings
            team: Team to use in battles
        """
        super().__init__(
            account_configuration, 
            avatar=avatar, 
            battle_format=battle_format, 
            log_level=log_level, 
            max_concurrent_battles=max_concurrent_battles, 
            accept_open_team_sheet=accept_open_team_sheet, 
            save_replays=save_replays, 
            server_configuration=server_configuration, 
            start_timer_on_battle_start=start_timer_on_battle_start, 
            start_listening=start_listening, 
            open_timeout=open_timeout, 
            ping_interval=ping_interval, 
            ping_timeout=ping_timeout, 
            team=team)
        
        self.do_data_collection = False  # Flag to control data collection
        self.uses_neural_network = False
        self.game_log = []  # Store battle data points
        
    def set_neural_network(self, neural_network):
        self.uses_neural_network = True
        self.neural_network = neural_network
    
    def set_data_collection(self, do_data_collection: bool):
        """
        Enable or disable data collection during battles.
        
        Args:
            do_data_collection: True to enable data collection, False to disable
        """
        self.do_data_collection = do_data_collection
    
    def choose_move(self, battle):
        """
        Choose a move for the current battle and optionally log the data.
        
        Args:
            battle: Current battle state
            
        Returns:
            The selected move to execute
        """
        move = self.choose_random_move(battle)

        # There are some moves that don't play well and I haven't figured out how to deal with them
        # so here we are
        if not hasattr(move, "order"):
            return move

        data = DataPoint(battle)

        if self.uses_neural_network:
            policy, value = self.neural_network.forward(data.get_input())

            moves = {

            }
            for move in battle.available_moves:
                moves[move] = policy[Move_Indices[move._id]].item(), policy[Move_Indices[move._id] + Num_Moves].item()
            for move in battle.available_switches:
                moves[move] = policy[Pokemon_Indices[move.name] + 2 * Num_Moves].item(), policy[Pokemon_Indices[move.name] + 2 * Num_Moves].item()
            
            move, tera = sample_with_temperature(moves, 0.20)

            move = self.create_order(move, terastallize=(tera == 1 and battle._can_tera))


        if self.do_data_collection and hasattr(move, "order"):
            data.set_move(move)
            self.game_log.append(DataPoint(battle))
        return move
    
    def _battle_finished_callback(self, battle):
        """
        Callback executed when a battle finishes.
        
        Args:
            battle: The battle that just finished
            
        Returns:
            The result of the parent class's battle finished callback
        """

        if self.do_data_collection:
            for data in self.game_log:
                # battle.won contributed by Brian
                data.set_reward(1 if battle.won else 0)

            self.game_log.clear()
            
        return super()._battle_finished_callback(battle)
