import poke_env
import asyncio
from Agent import Agent
import time
from PokemonData import Pokemon_Indices, Ability_Indices, Item_Indices, Move_Indices
from CustomDataset import PokemonData
from NeuralNetwork import Network

async def run_battle(n_battles=100):
    n = Network(PokemonData.get_tensor_length())
    first_player = Agent()
    first_player.set_data_collection(True)
    first_player.set_neural_network(n)
    second_player = Agent()
    
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

    print(f"Total battles: {n_battles}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    print(len(Pokemon_Indices.keys()))
    print(len(Ability_Indices.keys()))
    print(len(Item_Indices.keys()))
    print(len(Move_Indices.keys()))
    asyncio.run(run_battle(500))