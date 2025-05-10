import poke_env
from poke_env.player import Player
from copy import deepcopy
from CustomDataset import DataPoint, CustomDataset
import sys
import random
from PokemonData import Pokemon_Indices, Ability_Indices, Item_Indices, Move_Indices, Num_Pokemon, Num_Abilities, Num_Items, Num_Moves


def sample_with_temperature(d, temperature=1.0):
    """
    Sample a key from dictionary d where probability is proportional to value^(1/temperature).
    
    Args:
        d: Dictionary with keys and values in range (0, 1)
        temperature: Temperature parameter controlling randomness
                     - temperature > 0: Higher values make distribution more uniform
                     - temperature = 0: Deterministic, selects key with highest value
    
    Returns:
        A key from the dictionary
    """
    # Handle edge case - empty dictionary
    if not d:
        raise ValueError("Dictionary cannot be empty")
    
    # Special case: temperature = 0 (deterministic)
    if temperature == 0:
        return max(d.items(), key=lambda x: x[1])[0]
    
    # Calculate weights based on formula: value^(1/temperature)
    weights = {k: v ** (1 / temperature) for k, v in d.items()}
    
    # Calculate total weight
    total_weight = sum(weights.values())
    
    # Normalize weights
    normalized_weights = {k: w / total_weight for k, w in weights.items()}
    
    # Convert to list of (key, weight) for random.choices
    items = list(normalized_weights.items())
    keys, probs = zip(*items)
    
    # Sample based on calculated probabilities
    return random.choices(keys, weights=probs, k=1)[0]


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
    
    def set_data_collection(self, CDS: CustomDataset):
        """
        Enable or disable data collection during battles.
        
        Args:
            do_data_collection: True to enable data collection, False to disable
        """
        self.do_data_collection = True
        self.CDS = CDS
    
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
        # so this accounts for that case
        if not hasattr(move, "order"):
            return move

        data = DataPoint(battle)

        # (Move, Tera, Idx): (Prob)
        moves = {

        }

        if self.uses_neural_network:
            policy, value = self.neural_network.forward(data.get_input().to('cuda'))
            policy = policy.to('cpu')

            for move in battle.available_moves:
                moves[(move, False, Move_Indices[move._id])] = policy[Move_Indices[move._id]].item()
                if battle._can_tera:
                    moves[(move, True, Move_Indices[move._id] + Num_Moves)] = policy[Move_Indices[move._id] + Num_Moves].item()
            for move in battle.available_switches:
                moves[(move, False, Pokemon_Indices[move.name] + 2 * Num_Moves)] = policy[Pokemon_Indices[move.name] + 2 * Num_Moves].item()
        # Uniform probability when not using neural networks
        else:
            for move in battle.available_moves:
                moves[(move, False, Move_Indices[move._id])] = 1
                if battle._can_tera:
                    moves[(move, True, Move_Indices[move._id] + Num_Moves)] = 1
            for move in battle.available_switches:
                moves[(move, False, Pokemon_Indices[move.name] + 2 * Num_Moves)] = 1
        
        result = sample_with_temperature(moves, 0.4)

        move = self.create_order(result[0], terastallize=result[1])
        data.set_move(result[2])


        if self.do_data_collection:
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
                data.set_reward(1.0 if battle.won else 0.0)

            self.CDS.add_samples(self.game_log)
            self.game_log.clear()
            
        return super()._battle_finished_callback(battle)
