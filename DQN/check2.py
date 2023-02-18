import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse
import numpy as np
from collections import deque
import random
import time
import pygame
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.99)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()

print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.int64).to(device)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActionStateModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

        self.model = self.create_model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        #self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)

    def create_model(self):
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(600))  # width
        convh = conv2d_size_out(conv2d_size_out(400))  # height
        linear_input_size = convw * convh * 64
        fc = nn.Linear(linear_input_size, self.action_dim)

        model = nn.Sequential(
            conv1,
            nn.LeakyReLU(),
            conv2,
            nn.LeakyReLU(),
            nn.Flatten(),
            fc
        )
        return model

    def forward(self, state):
        return self.model(state.to(device))

    def get_action(self, state):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        with torch.no_grad():
            #state = torch.tensor(state, dtype=torch.float32).to(device)
            q_value = self.forward(state).cpu().detach().numpy()[0]
        if random.random() < self.epsilon:
            a = random.randint(0, self.action_dim - 1)
            print(a)
            return a
        return np.argmax(q_value)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_memory = np.zeros((self.state_dim, 600, 400))
        self.model = ActionStateModel(self.state_dim, self.action_dim).to(self.device)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim).to(self.device)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        states, actions, rewards, next_states, done = self.buffer.sample()
        q_values = self.model(states)
        q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_model(next_states).max(1)[0].detach()
        expected_q_values = rewards + (args.gamma * next_q_values * (1 - done))
        loss = nn.MSELoss()(q_values, expected_q_values)
        self.model.update(loss)

    def training(self, max_episodes=1000):
        for episode in range(max_episodes):
            state = self.env.reset()
            state = self.env.render()
            total_reward = 0
            done = False
            while not done:
                state = self.get_state(state)
                action = self.model.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                next_state = self.get_state(next_state)
                total_reward += reward
                self.buffer.put(state, action, reward, next_state, done)
                state = next_state
                if self.buffer.size() > args.batch_size:
                    self.replay()
                    self.target_update()
            print(f"Episode: {episode + 1}, total reward: {total_reward}")

    def get_state(self, obs):
        obs = obs.transpose((2, 0, 1))
        obs = np.asarray(obs, dtype=np.float32) / 255
        self.image_memory[:-1] = self.image_memory[1:]
        self.image_memory[-1:] = obs
        state = np.expand_dims(self.image_memory, axis=0)
        return torch.from_numpy(state).to(self.device)


def main():
    print( gym.__version__)
    env = gym.make('CartPole-v1', new_step_api=True, render_mode='single_rgb_array').unwrapped
    agent = Agent(env)
    agent.training(max_episodes=10000)

if __name__ == "__main__":
    main()