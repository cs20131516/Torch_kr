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
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.99)
parser.add_argument('--eps_min', type=float, default=0.05)

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

class DuelingDQNModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQNModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

        self.feature_model, self.value_model, self.advantage_model = self.create_model()
        self.feature_model.to(device)
        self.value_model.to(device)
        self.advantage_model.to(device)
        self.optimizer = optim.Adam(list(self.value_model.parameters()) + list(self.advantage_model.parameters()) + list(self.feature_model.parameters()), lr=args.lr)

    def create_model(self):
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(600))  # width
        convh = conv2d_size_out(conv2d_size_out(400))  # height
        linear_input_size = convw * convh * 64

        feature_model = nn.Sequential(
            conv1,
            nn.LeakyReLU(),
            conv2,
            nn.LeakyReLU(),
            nn.Flatten()
        )

        value_model = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 1)
        )

        advantage_model = nn.Sequential(
            nn.Linear(linear_input_size, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.action_dim)
        )

        return feature_model, value_model, advantage_model

    def forward(self, state):
        features = self.feature_model(state.to(device))
        values = self.value_model(features)
        advantages = self.advantage_model(features)
        q_values = values + (advantages - advantages.mean(dim=-1, keepdim=True))
        return q_values

    def get_action(self, state):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        with torch.no_grad():
            #state = torch.tensor(state, dtype=torch.float32).to(device)
            q_value = self.forward(state).cpu().detach().numpy()[0]
        if random.random() < self.epsilon:
            #a = random.randint(0, self.action_dim - 1)
            #print(a)
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)

    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

'''
위의 코드에서, 우선 `DuelingDQNModel` 클래스를 추가하였습니다. 
이 클래스는 피쳐 추출(feature extraction) 모델, 가치 모델(value model), 그리고 어드밴티지 모델(advantage model)로 구성되어 있습니다. 
이를 통해 Dueling DQN 아키텍처를 구현하였습니다.
그 다음, 기존 `Agent` 클래스에서 `ActionStateModel`을 `DuelingDQNModel`로 변경하였습니다. 나머지 코드는 기본 DQN과 동일하게 유지하였습니다. 
이를 통해 Dueling DQN 알고리즘을 사용하여 학습을 수행할 수 있습니다.
'''

class Agent:
    def __init__(self, env):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DuelingDQNModel(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DuelingDQNModel(self.state_dim, self.action_dim).to(self.device)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)
        self.ROWS = 600
        self.COLS = 400
        self.FRAME_STEP = 3
        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS))
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS)

    # 나머지 메소드는 동일하게 유지합니다.

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        #for _ in range(10):
        states, actions, rewards, next_states, done = self.buffer.sample()
        #states = torch.tensor(states, device=self.device, dtype=torch.float32).clone().detach()
        #actions = torch.tensor(actions, device=self.device, dtype=torch.int64).clone().detach()
        #rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).clone().detach()
        #next_states = torch.tensor(next_states, device=self.device, dtype=torch.float32).clone().detach()
        #done = torch.tensor(done, device=self.device, dtype=torch.float32).clone().detach()

        targets = self.target_model(states)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1).values
        targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
        loss = nn.MSELoss()(targets, self.model(states))
        #loss = nn.SmoothL1Loss()(targets, self.model(states))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def step(self, action):
        next_state, reward, done, _, _ = self.env.step(action)
        #next_state = self.GetImage()
        return next_state, reward, done

    def training(self, max_episodes=1000):
        start_time = time.time()
        end_time = 0
        epi_rewad = 0
        for ep in range(max_episodes):
            pygame.init()
            screen = pygame.display.set_mode((600, 400), pygame.HWSURFACE | pygame.DOUBLEBUF)
            done, total_reward = False, 0
            state = self.env.reset()
            state = np.array(pygame.surfarray.array3d(screen))
            state = state.transpose((2, 0, 1)) #/ 255.0
            epi_rewad = 0
            i = 0
            while not done:
                pygame.display.update()
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.model.get_action(state)
                #print(action)
                #print(i)
                _, reward, done = self.step(action)
                epi_rewad += reward
                #print(epi_rewad)
                next_state = np.array(pygame.surfarray.array3d(screen))
                next_state = next_state.transpose((2, 0, 1)) #/ 255.0

                if done:
                    if ep % 10 == 0:
                        self.target_update()

                self.buffer.put(state.squeeze().cpu().numpy(), action, reward, next_state, done)
                total_reward += reward
                state = next_state
                #if self.buffer.size() >= args.batch_size:
                    #self.replay()
                #if i % 50 == 0:
                    #self.target_update()
                #pygame.time.delay(10)#1000 -> 1second
                i += 1
            #if i < 50:
                #self.target_update()
            if self.buffer.size() >= args.batch_size:
                self.replay()
            print('EP{} EpisodeReward={} total_step={}'.format(ep, total_reward, i))
            f = open("DuelingDQN_Pic_epi_reward.txt", 'a')
            f.write(epi_rewad.__str__())
            f.write("\n")
            f.close()
            if epi_rewad >= 500:
                break
        end_time = time.time()
        f = open("DuelingDQN_Pic_complet_time.txt", 'a')
        f.write((end_time - start_time).__str__())
        f.write("\n")
        f.close()
        # Close the environment and pygame window
        torch.save(self.model.state_dict(), "./DuelingDQN_Pic_model.pt")
        self.env.close()
        pygame.quit()
        sys.exit()

def main():
    env = gym.make('CartPole-v1', new_step_api=True, render_mode='human').unwrapped
    agent = Agent(env)
    agent.training(max_episodes=10000)

if __name__ == "__main__":
    main()