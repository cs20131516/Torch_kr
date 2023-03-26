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
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.99)
parser.add_argument('--eps_min', type=float, default=0.05)
parser.add_argument('--clip_param', type=float, default=0.2)

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


class PPO_PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PPO_PolicyNetwork, self).__init__()
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

class PPO_Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PPO_PolicyNetwork(self.state_dim, self.action_dim).to(self.device)

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
            surr2 = torch.clamp(ratios, 1 - args.clip_param, 1 + args.clip_param) * advantages
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
        torch.save(self.policy.state_dict(), "./PPO_Discrete_Pic_model.pt")

        # Close the environment and pygame window
        self.env.close()
        pygame.quit()
        sys.exit()

def main():
    env = gym.make('CartPole-v1', new_step_api=True, render_mode='human').unwrapped
    agent = PPO_Agent(env)
    agent.training(max_episodes=10000)

if __name__ == "__main__":
    main()

'''
클래스 및 함수 이름을 PPO와 관련된 이름으로 변경했습니다.
PPO에 맞게 손실 함수를 변경했습니다.
delta를 clip_param으로 이름을 변경했습니다.


이제 위 코드를 실행하면 PPO 알고리즘을 사용하는 에이전트가 생성되고 학습을 진행합니다. 
학습이 완료되면 모델이 저장되고 환경이 종료됩니다.

코드의 변경 사항은 다음과 같습니다:

클래스 및 함수 이름을 PPO와 관련된 이름으로 변경했습니다. 
예를 들어, PolicyNetwork 클래스를 PPO_PolicyNetwork로 변경하였고, Agent 클래스를 PPO_Agent로 변경하였습니다.
손실 함수를 PPO에 맞게 변경했습니다. 
PPO에서는 목표 함수를 클리핑하여 계산합니다. 
이를 통해 업데이트의 크기를 제한하여 학습의 안정성을 향상시킵니다.
delta를 clip_param으로 이름을 변경했습니다. 
PPO의 클리핑 파라미터를 나타내기 위해 이름을 변경하였습니다.

TRPO와 PPO의 주요 차이점은 다음과 같습니다:

최적화 방법: TRPO는 자연 그래디언트 방법을 사용하여 목표 함수를 최적화하는 반면, PPO는 일반적인 경사하강법을 사용합니다. 
이로 인해 PPO는 구현이 간단하고, 계산 효율이 높습니다.

목표 함수의 제한: TRPO는 KL 발산이라는 척도를 사용하여 업데이트의 크기를 제한합니다. 
이를 통해 정책의 업데이트가 너무 크지 않도록 하여 안정성을 유지합니다. 
반면, PPO는 목표 함수를 클리핑하여 업데이트의 크기를 제한합니다. 
이 방법은 TRPO보다 더 간단하고 효율적입니다.

효율성 및 안정성: PPO는 TRPO보다 더 적은 계산량이 필요하고, 구현이 더 간단하기 때문에 효율성이 높습니다. 
또한, PPO는 TRPO와 유사한 성능을 보이면서 안정성이 높은 것으로 알려져 있습니다. 
이러한 이유로 PPO가 최근 많이 사용되는 알고리즘 중 하나입니다.
'''
