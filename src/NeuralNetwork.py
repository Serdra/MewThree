from CustomDataset import PokemonData, DataPoint
import torch
import torch.nn as nn


class Attention_Block(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, mlp_dim=384):
        super(Attention_Block, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim=embed_dim, heads=num_heads)
        self.up_proj = nn.Linear(embed_dim, mlp_dim)
        self.down_proj = nn.Linear(mlp_dim, embed_dim)
    
    def forward(self, input):
        activation = self.self_attn(input, input, input)

        activation = self.up_proj(activation)
        activation = nn.Mish()(activation)
        activation = self.down_proj(activation)

        return input + activation


class Network(nn.Module):
    def __init__(self, input_dim, embed_dim=256, num_heads=8, mlp_dim=384, num_moves=351, num_pokemon=420, down_proj_size=64):
        super(Network, self).__init__()

        self.embedding = nn.Linear(input_dim, embed_dim)
        self.positional_encoding = nn.Parameter(torch.randn(12, embed_dim))

        self.attention1 = Attention_Block(embed_dim, num_heads, mlp_dim)
        self.attention2 = Attention_Block(embed_dim, num_heads, mlp_dim)

        self.down_proj = nn.Linear(embed_dim, down_proj_size)

        self.policy = nn.Linear(down_proj_size * 12, num_moves * 2 + num_pokemon)

        self.value1 = nn.Linear(down_proj_size * 12, 32)
        self.value2 = nn.Linear(32, 1)
    
    def forward(self, x):
        """
        Forward pass for the neural network.
        
        Args:
            x: Input tensor of shape [batch_size, 12, input_dim]
                where 12 represents the 12 Pokémon (6 from each team)
        
        Returns:
            policy_output: Policy logits of shape [batch_size, num_moves * 2 + num_pokemon]
            value_output: Value prediction of shape [batch_size, 1]
        """
        batch_size = x.shape[0]
        
        # Apply embedding to each of the 12 Pokémon vectors
        # Input shape: [batch_size, 12, input_dim]
        # Output shape: [batch_size, 12, embed_dim]
        x = self.embedding(x)
        
        # Add positional encoding
        # The positional encoding should have shape [12, embed_dim]  
        # We need to ensure it's properly broadcasted to each item in the batch
        pos_encoding = self.positional_encoding[:12, :]  # Make sure dimensions match
        x = x + pos_encoding.unsqueeze(0)  # Add batch dimension for broadcasting
        
        # Apply attention blocks
        # Input and output shape: [batch_size, 12, embed_dim]
        x = self.attention1(x)
        x = self.attention2(x)
        
        # Apply down projection
        # Input shape: [batch_size, 12, embed_dim]
        # Output shape: [batch_size, 12, down_proj_size]
        x = self.down_proj(x)
        
        # Flatten the tensor for the final linear layers
        # Reshape from [batch_size, 12, down_proj_size] to [batch_size, 12 * down_proj_size]
        x_flat = x.reshape(batch_size, -1)
        
        # Policy head
        # Input shape: [batch_size, 12 * down_proj_size]
        # Output shape: [batch_size, num_moves * 2 + num_pokemon]
        policy_output = self.policy(x_flat)
        
        # Value head
        # Input shape: [batch_size, 12 * down_proj_size]
        # Through value1: [batch_size, 32]
        # Final output: [batch_size, 1]
        value_hidden = nn.Mish()(self.value1(x_flat))
        value_output = torch.sigmoid(self.value2(value_hidden))
        
        return policy_output, value_output