import poke_env
import asyncio
from Agent import Agent
import time
from PokemonData import Pokemon_Indices, Ability_Indices, Item_Indices, Move_Indices
from CustomDataset import PokemonData, CustomDataset
from NeuralNetwork import Network

async def run_battle(n_battles=100):
    n = Network(PokemonData.get_tensor_length()).to('cuda')
    CDS = CustomDataset(50000)
    first_player = Agent()
    second_player = Agent()
    first_player.set_data_collection(CDS)
    first_player.set_neural_network(n)
    
    # Start timing
    start_time = time.time()

    # The battle_against method initiates a battle between two players.
    # Here we are using asynchronous programming (await) to start the battle.
    await first_player.battle_against(second_player, n_battles=n_battles)
    
    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    # Calculate battles per second
    battles_per_second = n_battles / elapsed_time
    
    print(f"Neural player won {first_player.n_won_battles} battles")
    print(f"Total battles: {n_battles}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Samples gathered: {CDS.samples_since_last_step}")

if __name__ == "__main__":
    asyncio.run(run_battle(500))