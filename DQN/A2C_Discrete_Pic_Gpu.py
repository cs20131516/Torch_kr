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
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--entropy_coef', type=float, default=0.01)

args = parser.parse_args()

print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.model = self.create_model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

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
            nn.Softmax(dim=-1)
        )
        return model

    def forward(self, state):
        return self.model(state.to(device))

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.state_dim = state_dim

        self.model = self.create_model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def create_model(self):
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(600))  # width
        convh = conv2d_size_out(conv2d_size_out(400))  # height
        linear_input_size = convw * convh * 64
        fc = nn.Linear(linear_input_size, 1)

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

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.critic = Critic(self.state_dim).to(self.device)

        self.ROWS = 600
        self.COLS = 400
        self.FRAME_STEP = 3
        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS))
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS)

    def step(self, action):
        next_state, reward, done, _, _ = self.env.step(action)
        return next_state, reward, done

    def update(self, log_probs, values, rewards, dones):
        returns = []
        R = 0
        for r, d in zip(rewards[::-1], dones[::-1]):
            R = r + args.gamma * R * (1 - d)
            returns.insert(0, R)
        returns = torch.tensor(returns).to(self.device)

        advantages = returns - values

        actor_loss = (-log_probs * advantages.detach()).mean()
        critic_loss = advantages.pow(2).mean()

        self.actor.optimizer.zero_grad()
        self.critic.optimizer.zero_grad()

        loss = actor_loss + critic_loss - args.entropy_coef * (-(log_probs.exp() * log_probs).mean())
        loss.backward()

        self.actor.optimizer.step()
        self.critic.optimizer.step()

    def training(self, max_episodes=1000):
        start_time = time.time()
        end_time = 0
        epi_reward = 0
        for ep in range(max_episodes):
            pygame.init()
            screen = pygame.display.set_mode((600, 400), pygame.HWSURFACE | pygame.DOUBLEBUF)
            done, total_reward = False, 0
            state = self.env.reset()
            state = np.array(pygame.surfarray.array3d(screen))
            state = state.transpose((2, 0, 1))
            epi_reward = 0
            i = 0

            log_probs = []
            values = []
            rewards = []
            dones = []

            while not done:
                pygame.display.update()
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_probs = self.actor(state)
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                value = self.critic(state)
                _, reward, done = self.step(action.item())

                epi_reward += reward
                next_state = np.array(pygame.surfarray.array3d(screen))
                next_state = next_state.transpose((2, 0, 1))

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                dones.append(done)

                total_reward += reward
                state = next_state
                i += 1

            self.update(torch.cat(log_probs), torch.cat(values), rewards, dones)
            print('EP{} EpisodeReward={} total_step={}'.format(ep, total_reward, i))

            if epi_reward >= 500:
                break
            f = open("A2C_Pic_epi_reward.txt", 'a')
            f.write(epi_reward.__str__())
            f.write("\n")
            f.close()

        end_time = time.time()
        f = open("A2C_Pic_complet_time.txt", 'a')
        f.write((end_time - start_time).__str__())
        f.write("\n")
        f.close()

        # Save the model
        torch.save(self.actor.state_dict(), "./A2C_Pic_actor_model.pt")
        torch.save(self.critic.state_dict(), "./A2C_Pic_critic_model.pt")

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
이제 코드는 A2C 방법을 사용하여 학습합니다. 주요 변경 사항은 다음과 같습니다.

Agent 클래스에서 DQN 관련 코드를 제거하고 A2C 관련 코드를 추가했습니다.
Agent 클래스의 training() 함수에서 에피소드별로 actor 및 critic 네트워크를 업데이트하는 과정을 추가했습니다.
훈련이 완료된 후에 actor 및 critic 모델을 저장합니다.
결과를 A2C에 대한 파일로 저장하도록 파일 이름을 변경했습니다.
'''