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
from memory_profiler import profile
import gc

torch.serialization.add_safe_globals([Network])

# Updated to account for a global event loop
def train_network():
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
    
    # Create a single event loop that will be reused throughout the function
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        neural_player = Agent(max_concurrent_battles=1)
        neural_player.set_data_collection(CDS)

        neural_player_2 = poke_env.MaxBasePowerPlayer(max_concurrent_battles=1)

        neural_player.temperature = 0.1

        # Prefills the dataset with 500 games
        neural_player.reset_battles()
        
        # Instead of asyncio.run which creates and destroys loops
        loop.run_until_complete(neural_player.battle_against(neural_player_2, n_battles=250))
        print(f"Learning player won {neural_player.n_won_battles}")

        neural_player.set_neural_network(n1)
        neural_player.temperature = 0.4

        # Trains quickly
        Train(n1, CDS, optimizer, 16, 256, 'cuda')

        CDS.samples_since_last_step = 0
        num_steps = 0

        # Regularly clean up memory and pending tasks
        gc_counter = 0
        
        # Learns for 15 minutes
        start_time = time.time()
        while True:
            with torch.no_grad():
                # Run battles in chunks with proper cleanup
                battle_task = loop.run_until_complete(
                    neural_player.battle_against(neural_player_2, n_battles=25)
                )
                
            # Reset battles and clear any accumulated state
            neural_player.reset_battles()
            neural_player_2.reset_battles()

            # Process accumulated samples in batches
            while CDS.samples_since_last_step > 256:
                CDS.samples_since_last_step -= 256
                Train(n1, CDS, optimizer, 2, 256, 'cuda')
            
                if num_steps % 100 == 0 and num_steps != 0:
                    print(f"Steps: {num_steps}")
                    print(f"Elapsed: {time.time() - start_time:.2f} seconds\n")
                
                if num_steps % 500 == 0 and num_steps != 0:
                    torch.save(n1.state_dict(), f"net_{num_steps}.pt")
                    
                num_steps += 1
                
            # Periodically clean up memory and pending tasks
            gc_counter += 1
            if gc_counter % 10 == 0:
                # Clear any pending tasks in the event loop
                for task in asyncio.all_tasks(loop):
                    if not task.done():
                        task.cancel()
                
                # Force garbage collection
                gc.collect()
                
                # Print memory stats to help debug
                if gc_counter % 50 == 0:
                    print(f"Active tasks: {len(asyncio.all_tasks(loop))}")
                    
    finally:
        # Ensure proper cleanup when the function exits
        pending = asyncio.all_tasks(loop)
        for task in pending:
            task.cancel()
        
        # Run the loop until all tasks are canceled
        if pending:
            loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            
        loop.close()

    
if __name__ == "__main__":
    train_network()