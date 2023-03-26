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


class ActionStateModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActionStateModel, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = args.eps

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
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        with torch.no_grad():
            q_value = self.forward(state).cpu().detach().numpy()[0]
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = ActionStateModel(self.state_dim, self.action_dim).to(self.device)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim).to(self.device)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)
        self.ROWS = 600
        self.COLS = 400
        self.FRAME_STEP = 3
        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS))
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS)


    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())


    def replay(self):
        states, actions, rewards, next_states, done = self.buffer.sample()

        # Double DQN update
        model_next_actions = self.model(next_states).argmax(dim=1)
        with torch.no_grad():
            target_next_q_values = self.target_model(next_states).gather(1, model_next_actions.unsqueeze(1)).squeeze()
        targets = self.target_model(states)
        targets[range(args.batch_size), actions] = rewards + (1 - done) * target_next_q_values * args.gamma

        loss = nn.MSELoss()(targets, self.model(states))
        self.model.update(loss)

    '''
    코드에서 바뀐 부분은 Agent 클래스의 replay() 메소드입니다. 이 부분에서 기존 DQN의 업데이트 방식을 Double DQN으로 변경했습니다.
    
    DQN에서는 다음 Q-값을 추정할 때 동일한 네트워크를 사용하여 가장 큰 값을 선택합니다. 이는 오버 에스티메이션 문제를 초래할 수 있습니다. 오버 에스티메이션 문제란 추정된 Q-값이 실제 Q-값보다 높을 때 발생하는 문제입니다.
    
    Double DQN은 이 오버 에스티메이션 문제를 완화하기 위해 두 개의 네트워크를 사용합니다: 하나는 현재 학습 중인 네트워크(온라인 네트워크)이고, 다른 하나는 일정 주기로 업데이트되는 타겟 네트워크입니다. 타겟 네트워크는 온라인 네트워크의 가중치를 복사하여 업데이트됩니다. 이렇게 두 개의 네트워크를 사용함으로써 추정 오류를 줄일 수 있습니다.
    
    Double DQN에서의 차이점은 다음과 같습니다:
    
    온라인 네트워크를 사용하여 다음 상태에서 최적의 행동을 선택합니다 (model_next_actions).
    타겟 네트워크를 사용하여 다음 상태에서 선택된 행동의 Q-값을 계산합니다 (target_next_q_values).
    이러한 변경 사항은 오버 에스티메이션 문제를 완화하고, 훈련 과정에서 더 안정적인 학습을 가능하게 합니다.
    '''
    def step(self, action):
        next_state, reward, done, _, _ = self.env.step(action)
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
            state = state.transpose((2, 0, 1))
            epi_rewad = 0
            i = 0
            while not done:
                pygame.display.update()
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = self.model.get_action(state)
                _, reward, done = self.step(action)
                epi_rewad += reward
                next_state = np.array(pygame.surfarray.array3d(screen))
                next_state = next_state.transpose((2, 0, 1))

                if done:
                    if ep % 10 == 0:
                        self.target_update()

                self.buffer.put(state.squeeze().cpu().numpy(), action, reward, next_state, done)
                total_reward += reward
                state = next_state
                i += 1

            if self.buffer.size() >= args.batch_size:
                self.replay()

            print('EP{} EpisodeReward={} total_step={}'.format(ep, total_reward, i))
            f = open("DoubleDQN_Pic_epi_reward.txt", 'a')
            f.write(epi_rewad.__str__())
            f.write("\n")
            f.close()

            if epi_rewad >= 500:
                break

        end_time = time.time()
        f = open("DoubleDQN_Pic_complet_time.txt", 'a')
        f.write((end_time - start_time).__str__())
        f.write("\n")
        f.close()

        # Save the trained model
        torch.save(self.model.state_dict(), "./DoubleDQN_Pic_model.pt")

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