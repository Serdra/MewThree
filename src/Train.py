import torch
from NeuralNetwork import Network
from CustomDataset import CustomDataset

def Train(network, dataset, optimizer, num_batches, batch_size=32, device='cuda'):
    """
    Train a neural network using SGD with Nesterov momentum.
    
    Args:
        network: The neural network model to train
        dataset: CustomDataset instance for sampling training data
        num_batches: Number of batches to train on (not epochs)
        batch_size: Number of samples per batch (default: 32)
        learning_rate: Learning rate for optimizer (default: 0.001)
        momentum: Momentum coefficient (default: 0.9)
        nesterov: Whether to use Nesterov momentum (default: True)
        value_coef: Coefficient for value loss (default: 1.0)
        device: Device to train on ('cuda' or 'cpu', default: 'cuda')
    
    Returns:
        dict: Dictionary containing average policy loss and value loss
    """
    # Move network to device
    network = network.to(device)
    network.train()
    
    # Loss tracking
    total_policy_loss = 0.0
    total_value_loss = 0.0
    
    # Train for specified number of batches
    for batch_idx in range(num_batches):
        # Initialize batch tensors
        batch_inputs = []
        batch_policy_indices = []
        batch_rewards = []
        
        # Fill batch with samples
        for _ in range(batch_size):
            input_data, policy_index, reward = dataset.get_sample()
            batch_inputs.append(input_data)
            batch_policy_indices.append(policy_index)
            batch_rewards.append(reward)
        
        # Convert to tensors and move to device
        batch_inputs = torch.stack(batch_inputs).to(device)
        
        # Verify correct shape
        if len(batch_inputs.shape) != 3 or batch_inputs.shape[1] != 12:
            raise ValueError(f"Expected input shape (batch_size, 12, input_dim), got {batch_inputs.shape}")
        batch_policy_indices = torch.tensor(batch_policy_indices, dtype=torch.long).to(device)
        batch_rewards = torch.tensor(batch_rewards, dtype=torch.float32).to(device).unsqueeze(1)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        policy_output, value_output = network(batch_inputs)
        
        # Compute policy loss - only for specified index for each sample
        # For Q-learning, we use MSE between policy output and reward
        policy_loss = 0
        for i in range(batch_size):
            # Only use the specified policy index for loss calculation
            # The policy outputs are trained towards the same reward values
            target = batch_rewards[i]
            prediction = policy_output[i, batch_policy_indices[i]].unsqueeze(0)
            policy_loss += torch.nn.functional.mse_loss(prediction, target)
        
        # Compute value loss
        value_loss = torch.nn.functional.mse_loss(value_output, batch_rewards)
        
        # Combine losses
        total_loss = policy_loss + value_loss
        
        # Backward pass
        total_loss.backward()
        
        # Update weights
        optimizer.step()
        
        # Track losses
        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        
        # Print progress every 100 batches
        if (batch_idx + 1) % 100 == 0:
            print(f"Batch {batch_idx + 1}/{num_batches}, "
                  f"Policy Loss: {total_policy_loss / (batch_idx + 1):.4f}, "
                  f"Value Loss: {total_value_loss / (batch_idx + 1):.4f}")
    
    network.eval()
    # Return average losses
    return {
        'policy_loss': total_policy_loss / num_batches,
        'value_loss': total_value_loss / num_batches
    }