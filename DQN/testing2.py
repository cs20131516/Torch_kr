import gym
import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# Define the neural network
class Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Initialize the environment
env = gym.make('CartPole-v1', new_step_api=True, render_mode='human').unwrapped

# Initialize the neural network
input_size = 4
output_size = 2
policy_net = Net(input_size, output_size)

# Initialize the optimizer
optimizer = optim.SGD(policy_net.parameters(), lr=0.01)

# Initialize the loss function
criterion = nn.MSELoss()

# Initialize the pygame screen
pygame.init()
screen = pygame.display.set_mode((400, 300))

# Train the agent
for episode in range(100):
    # Reset the environment
    state = env.reset()
    frame_count = 0

    # Initialize the cumulative reward
    cum_reward = 0

    # Initialize the episode
    done = False

    while not done:
        frame_count += 1

        # Get the pygame screen every 4 frames
        if frame_count % 4 == 0:
            state = np.array(pygame.surfarray.array3d(screen))
            state = state.transpose((2, 0, 1)) / 255.0
            state = state.astype(np.float32).reshape(-1)

        # Pass the state through the policy network
        state_tensor = torch.tensor([state], dtype=torch.float32)
        action_probs = policy_net(state_tensor)

        # Take the action with the highest probability
        action = torch.argmax(action_probs).item()

        # Take a step in the environment
        next_state, reward, done, _, _ = env.step(action)

        # Compute the loss
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        expected_reward = reward_tensor + 0.99 * torch.max(policy_net(torch.tensor([next_state], dtype=torch.float32)))
        expected_action_probs = torch.zeros_like(action_probs)
        expected_action_probs[0][action] = expected_reward
        loss = criterion(action_probs, expected_action_probs)

        # Update the cumulative reward
        cum_reward += reward

        # Update the policy network
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the state
        state = next_state

    # Close the environment and pygame window
    env.close()
    pygame.quit()
