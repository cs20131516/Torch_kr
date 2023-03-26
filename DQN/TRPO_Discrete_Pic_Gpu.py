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
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.99)
parser.add_argument('--eps_min', type=float, default=0.05)
parser.add_argument('--delta', type=float, default=0.01)

args = parser.parse_args()

print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done, log_prob):
        self.buffer.append([state, action, reward, next_state, done, log_prob])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done, log_probs = map(np.asarray, zip(*sample))
        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.int64).to(device)
        log_probs = torch.tensor(log_probs, dtype=torch.float32).to(device)
        return states, actions, rewards, next_states, done, log_probs

    def size(self):
        return len(self.buffer)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNetwork, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

        self.model = self.create_model().to(device)

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
            fc,
            nn.Softmax(dim=1)
        )
        return model

    def forward(self, state):
        return self.model(state.to(device))

    def get_action(self, state):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        with torch.no_grad():
            state = state.to(device)
        prob = self.forward(state)
        dist = Categorical(prob)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(self.state_dim, self.action_dim).to(self.device)

        self.buffer = ReplayBuffer()
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.ROWS = 600
        self.COLS = 400
        self.FRAME_STEP = 3
        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS))
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS)

    def step(self, action):
        next_state, reward, done, _, _ = self.env.step(action)
        return next_state, reward, done

    def compute_advantages(self, rewards, dones):
        advantages = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            if dones[t]:
                running_add = 0
            running_add = running_add * args.gamma + rewards[t]
            advantages[t] = running_add
        return advantages

    def optimize_policy(self, states, actions, advantages, log_probs_old):
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.int64).to(self.device)
        advantages = torch.tensor(np.array(advantages), dtype=torch.float32).to(self.device)
        log_probs_old = torch.tensor(np.array(log_probs_old), dtype=torch.float32).to(self.device)

        for _ in range(10):
            prob = self.policy(states)
            dist = Categorical(prob)
            log_probs_new = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            ratios = torch.exp(log_probs_new - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - args.delta, 1 + args.delta) * advantages
            loss = -torch.min(surr1, surr2).mean() - 0.001 * entropy
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def training(self, max_episodes=1000):
        for ep in range(max_episodes):
            pygame.init()
            screen = pygame.display.set_mode((600, 400), pygame.HWSURFACE | pygame.DOUBLEBUF)
            done, total_reward = False, 0
            state = self.env.reset()
            state = np.array(pygame.surfarray.array3d(screen))
            state = state.transpose((2, 0, 1))
            states, actions, rewards, dones, log_probs = [], [], [], [], []

            while not done:
                pygame.display.update()
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, log_prob = self.policy.get_action(state)
                next_state, reward, done = self.step(action)

                states.append(state.squeeze().cpu().numpy())
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                log_probs.append(log_prob.cpu().detach().numpy())

                total_reward += reward
                state = np.array(pygame.surfarray.array3d(screen))
                state = state.transpose((2, 0, 1))

            advantages = self.compute_advantages(rewards, dones)
            self.optimize_policy(states, actions, advantages, log_probs)

            print('EP{} EpisodeReward={}'.format(ep, total_reward))

            if total_reward >= 500:
                break

        # Save the trained model
        torch.save(self.policy.state_dict(), "./TRPO_Discrete_Pic_model.pt")

        # Close the environment and pygame window
        self.env.close()
        pygame.quit()
        sys.exit()

def main():
    env = gym.make('CartPole-v1', new_step_api=True, render_mode='human').unwrapped
    agent = Agent(env)
    agent.training(max_episodes=10000)

if __name__ == "__main__":
    main()

'''
알고리즘 유형:

DQN (Deep Q-Network)은 가치 기반(value-based) 강화학습 알고리즘이며, 상태-행동 쌍에 대한 Q-값을 예측하기 위해 신경망을 사용합니다. 
DQN은 이러한 Q-값을 최적화하여 최적의 정책을 찾으려고 합니다.
TRPO (Trust Region Policy Optimization)은 정책 기반(policy-based) 강화학습 알고리즘이며, TRPO는 직접적으로 정책을 최적화하는 데 초점을 맞춥니다. 
TRPO는 KL 발산(Kullback-Leibler divergence)이라는 개념을 사용하여 이전 정책과 새 정책 사이의 변화를 제한하며 최적화를 수행합니다.

최적화 방식:

DQN은 Q-값의 차이를 최소화하는 방향으로 신경망의 가중치를 조정하여 최적화를 수행합니다. 
DQN은 고정 타겟 네트워크와 에피소드 또는 일정 시간마다 업데이트되는 메인 네트워크를 사용하여 안정성을 향상시킵니다.
TRPO는 목적 함수(objective function)를 최적화하면서 KL 발산을 제한하여 정책을 최적화합니다. 
이 과정에서 TRPO는 연속 및 이산 동작 공간에서 모두 적용할 수 있는 것으로 입증되었습니다.
이 코드에서는 Agent 클래스와 PolicyNetwork 클래스를 사용하여 TRPO 알고리즘을 구현하고 있습니다. 
PolicyNetwork는 환경에서 얻은 상태를 기반으로 정책을 예측하며, Agent 클래스는 환경과 상호 작용하고 최적화를 수행합니다.
'''