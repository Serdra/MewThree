import os

# Global dictionaries mapping Pokemon game elements to numerical indices
Pokemon_Indices = {}  # Maps Pokemon names to unique indices
Ability_Indices = {}  # Maps ability names to unique indices
Move_Indices = {}     # Maps move names to unique indices
Item_Indices = {}     # Maps item names to unique indices

def _read_pokemon_data_from_files():
    """
    Load Pokemon game data from text files and populate global index dictionaries.
    
    Each element (Pokemon, ability, move, item) is assigned a unique positive integer ID.
    None values are assigned index 0 for each category.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Read Pokemon data
    Pokemon_Indices[None] = 0  # Reserve 0 for None/unknown Pokemon
    Pokemon_Indices[""] = 0
    pokemon_path = os.path.join(script_dir, "data", "pokemon.txt")
    with open(pokemon_path, "r") as file:
        for i, line in enumerate(file, 1):  # Start indices from 1
            pokemon_name = line.strip()
            if pokemon_name:  # Skip empty lines

                # The library isn't always consistent on whether it uses lower or upper case names, so here's both
                Pokemon_Indices[pokemon_name] = i
    
    # Read Ability data
    Ability_Indices[None] = 0  # Reserve 0 for None/unknown ability
    Ability_Indices[""] = 0
    ability_path = os.path.join(script_dir, "data", "abilities.txt")
    with open(ability_path, "r") as file:
        for i, line in enumerate(file, 1):  # Start indices from 1
            ability_name = line.strip()
            if ability_name:  # Skip empty lines
                Ability_Indices[ability_name] = i
    
    # Read Move data
    Move_Indices[None] = 0  # Reserve 0 for None/unknown move
    Move_Indices[""] = 0
    move_path = os.path.join(script_dir, "data", "moves.txt")
    with open(move_path, "r") as file:
        for i, line in enumerate(file, 1):  # Start indices from 1
            move_name = line.strip()
            if move_name:  # Skip empty lines
                Move_Indices[move_name] = i
    
    # Read Item data
    Item_Indices[None] = 0  # Reserve 0 for None/unknown item
    Item_Indices[""] = 0
    item_path = os.path.join(script_dir, "data", "items.txt")
    with open(item_path, "r") as file:
        for i, line in enumerate(file, 1):  # Start indices from 1
            item_name = line.strip()
            if item_name:  # Skip empty lines
                Item_Indices[item_name] = i

# Execute the function to load all data when module is imported
_read_pokemon_data_from_files()

Num_Pokemon = max(Pokemon_Indices.values())
Num_Abilities = max(Ability_Indices.values())
Num_Moves = max(Move_Indices.values())
Num_Items = max(Item_Indices.values())