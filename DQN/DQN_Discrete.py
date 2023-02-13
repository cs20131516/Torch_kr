import torch
import torch.nn as nn
import torch.optim as optim
import gym
import argparse
import numpy as np
from collections import deque
import random
import time

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()


class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.int64)
        return states, actions, rewards, next_states, done

    def size(self):
        return len(self.buffer)


class ActionStateModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActionStateModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

        self.model = self.create_model()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_dim)
        )
        return model

    def forward(self, state):
        return self.model(state)

    def get_action(self, state):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0)
            q_value = self.forward(state).detach().numpy()[0]
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
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

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model(states)
            with torch.no_grad():
                next_q_values = self.target_model(next_states).max(dim=1).values
            targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
            loss = nn.MSELoss()(targets, self.model(states))
            self.model.update(loss)

    def training(self, max_episodes=1000):
        start_time = time.time()
        end_time = 0
        epi_rewad = 0
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            epi_rewad = 0
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, _, _= self.env.step(action)
                epi_rewad += reward
                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
            if self.buffer.size() >= args.batch_size:
                self.replay()
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            f = open("DQN_Pic_epi_reward.txt", 'a')
            f.write(epi_rewad.__str__())
            f.write("\n")
            f.close()
        end_time = time.time()
        f = open("DQN_Pic_complet_time.txt", 'a')
        f.write((end_time - start_time).__str__())
        f.write("\n")
        f.close()

def main():
    env = gym.make('CartPole-v1', new_step_api=True, render_mode='human').unwrapped
    agent = Agent(env)
    agent.training(max_episodes=1000)

if __name__ == "__main__":
    main()