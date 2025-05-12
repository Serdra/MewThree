import poke_env
import asyncio
from Agent import Agent
import time
import torch
from PokemonData import Pokemon_Indices, Ability_Indices, Item_Indices, Move_Indices
from CustomDataset import PokemonData, CustomDataset
from NeuralNetwork import Network
from Train import Train
import cProfile

torch.serialization.add_safe_globals([Network])

async def test_network(n_battles=1000):
    n1 = Network(PokemonData.get_tensor_length())
    # n2 = Network(PokemonData.get_tensor_length())

    n1.load_state_dict(torch.load("C:\\Users\\serdr\\Desktop\\Programming\\MewThree\\net_0.pt", weights_only=False))
    # n2.load_state_dict(torch.load("C:\\Users\\serdr\\Desktop\\Programming\\MewThree\\opp_net.pt", weights_only=False))

    n1.to('cuda')
    # n2.to('cuda')

    p1 = Agent()
    p2 = poke_env.MaxBasePowerPlayer()

    p1.set_neural_network(n1)
    # p2.set_neural_network(n2)

    p1.temperature = 0.1
    # p2.temperature = 0.1

    await p1.battle_against(p2, n_battles=n_battles)

    print(f"Player 1 won {p1.n_won_battles} times")
    print(f"Player 2 won {p2.n_won_battles} times")
    print(f"They tied {p1.n_tied_battles} times")

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

async def train_network():
    n1 = Network(PokemonData.get_tensor_length()).to('cuda')

    torch.save(n1.state_dict(), "net_0.pt")

    CDS = CustomDataset(200000)
    optimizer = torch.optim.SGD(
        n1.parameters(),
        lr=1e-4,
        momentum=0.9,
        nesterov=True,
        weight_decay=1e-6
    )
    
    neural_player = Agent(max_concurrent_battles=1)
    neural_player.set_data_collection(CDS)

    neural_player_2 = poke_env.MaxBasePowerPlayer(max_concurrent_battles=1)

    neural_player.temperature = 0.1

    # Prefills the dataset with 500 games
    neural_player.reset_battles()
    await neural_player.battle_against(neural_player_2, n_battles=250)
    print(f"Learning player won {neural_player.n_won_battles}")

    neural_player.set_neural_network(n1)

    neural_player.temperature = 0.4

    # Trains quickly
    Train(n1, CDS, optimizer, 16, 256, 'cuda')

    CDS.samples_since_last_step = 0

    num_steps = 0

    # Learns for 15 minutes
    start_time = time.time()
    while True:
        with torch.no_grad():
            await neural_player.battle_against(neural_player_2, n_battles=25)
        neural_player.reset_battles()
        neural_player_2.reset_battles()

        while CDS.samples_since_last_step > 256:
            CDS.samples_since_last_step -= 256
            Train(n1, CDS, optimizer, 2, 256, 'cuda')
        
            if num_steps % 100 == 0 and num_steps != 0:
                print(f"Steps: {num_steps}")
                print(f"Elapsed: {time.time() - start_time:.2f} seconds\n")
            
            if num_steps % 500 == 0 and num_steps != 0:
                torch.save(n1.state_dict(), f"net_{num_steps}.pt")
                
            num_steps += 1
    
    
    neural_player.temperature = 0.1

    neural_player.reset_battles()
    await neural_player.battle_against(neural_player_2, n_battles=500)
    print(f"Learning player won {neural_player.n_won_battles}")
    
if __name__ == "__main__":
    asyncio.run(train_network())