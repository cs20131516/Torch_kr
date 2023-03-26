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
parser.add_argument('--tau', type=float, default=0.01)
parser.add_argument('--entropy_alpha', type=float, default=0.2)

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


class SoftActorCriticDiscrete(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(SoftActorCriticDiscrete, self).__init__()
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
            fc
        )
        return model

    def forward(self, state):
        return self.model(state.to(device))

    def get_action(self, state):
        with torch.no_grad():
            logits = self.forward(state).cpu().detach().numpy()[0]
            probs = self._probs_from_logits(logits)
            action = np.random.choice(range(self.action_dim), p=probs)
        return action

    def _probs_from_logits(self, logits):
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        return probs

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class PPO_Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SoftActorCriticDiscrete(self.state_dim, self.action_dim).to(self.device)
        self.target_model = SoftActorCriticDiscrete(self.state_dim, self.action_dim).to(self.device)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def target_update(self):
        for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_((1 - args.tau) * target_param.data + args.tau * param.data)

    def replay(self):
        states, actions, rewards, next_states, done = self.buffer.sample()

        logits = self.model(states)
        probs = torch.softmax(logits, dim=1)
        log_probs = torch.log_softmax(logits, dim=1)

        target_logits = self.target_model(next_states).detach()
        target_log_probs = torch.log_softmax(target_logits, dim=1)
        target_entropy = -torch.sum(torch.exp(target_log_probs) * target_log_probs, dim=1)

        values = (probs * log_probs).sum(dim=1)
        target_values = (1 - done) * (rewards + args.gamma * args.entropy_alpha * target_entropy)

        loss = -(values * target_values).mean()
        self.model.update(loss)

    def step(self, action):
        next_state, reward, done, _, _ = self.env.step(action)
        return next_state, reward, done

    def training(self, max_episodes=1000):
        for ep in range(max_episodes):
            pygame.init()
            screen = pygame.display.set_mode((600, 400), pygame.HWSURFACE | pygame.DOUBLEBUF)
            done, total_reward = False, 0
            state = self.env.reset()
            state = np.array(pygame.surfarray.array3d(screen))
            state = state.transpose((2, 0, 1))

            while not done:
                pygame.display.update()
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.model.get_action(state)
                next_state, reward, done = self.step(action)

                next_state = np.array(pygame.surfarray.array3d(screen))
                next_state = next_state.transpose((2, 0, 1))

                self.buffer.put(state.squeeze().cpu().numpy(), action, reward, next_state, done)
                total_reward += reward
                state = next_state

                if self.buffer.size() >= args.batch_size:
                    self.replay()
                    self.target_update()

                if total_reward >= 500:
                    break

            print('EP{} EpisodeReward={}'.format(ep, total_reward))
        torch.save(self.model.state_dict(), f"./SAC_Discrete_model_ep{ep}.pt")
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
코드의 변경된 부분은 다음과 같습니다:

SoftActorCriticDiscrete 클래스를 추가했습니다. 이 클래스는 이산 동작 공간에 대한 Soft Actor-Critic (SAC) 에이전트를 정의합니다.

Agent 클래스의 모델 및 목표 모델 초기화에서 DQN에서 SAC로 변경되었습니다.

target_update 함수를 변경하여 일부 가중치를 현재 모델로부터 목표 모델로 전달하도록 수정하였습니다.

replay 함수를 수정하여 SAC 학습 방법에 따라 손실을 계산하고 업데이트하도록 변경하였습니다.

훈련 루프에서 self.target_update()를 호출하는 부분을 더 자주 호출하도록 변경하였습니다.

SAC와 DQN의 차이점은 다음과 같습니다:

알고리즘: DQN은 Q-learning 알고리즘을 기반으로 하며, SAC는 Actor-Critic 방식을 사용합니다. 이 두 알고리즘은 강화 학습의 두 가지 주요 방법입니다.

탐색: DQN은 엡실론-그리디 탐색 전략을 사용하여 확률적으로 무작위 동작을 선택합니다. SAC는 확률적 정책을 사용하여 탐색과 개발을 자연스럽게 균형을 이룹니다.

손실 함수: DQN은 예측 Q 값과 목표 Q 값 사이의 평균 제곱 오차 (MSE)를 최소화하려고 합니다. SAC에서는 정책의 로그 확률과 목표 Q 값 사이의 교차 엔트로피를 최소화합니다.

엔트로피 정규화: SAC는 엔트로피 정규화를 사용하여 정책의 확률 분포를 평평하게 유지하려고 합니다. 이는 탐색 능력을 개선하고 학습 안정성을 높입니다. DQN에서는 이러한 메커니즘이 없습니다.
'''