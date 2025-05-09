import poke_env
from poke_env.player import Player
from copy import deepcopy
from dataformat import DataPoint
import sys


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
        self.game_log = []  # Store battle data points
        
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
        # Random move for now, this will be improved later
        move = self.choose_random_move(battle)

        if self.do_data_collection:
            self.game_log.append(DataPoint(battle, move))
        return move
    
    def _battle_finished_callback(self, battle):
        """
        Callback executed when a battle finishes.
        
        Args:
            battle: The battle that just finished
            
        Returns:
            The result of the parent class's battle finished callback
        """
        # Print how large the array is and how much memory it's taking up
        if self.do_data_collection:
            self.game_log.clear()
            
        return super()._battle_finished_callback(battle)
