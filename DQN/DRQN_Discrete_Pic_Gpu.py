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
parser.add_argument('--sequence_length', type=int, default=5, help="Number of steps in a sequence.")

args = parser.parse_args()

print(torch.cuda.is_available())
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class ReplayBuffer:
    def __init__(self, capacity=2000):
        self.buffer = deque(maxlen=capacity)

    def put(self, state, action, reward, next_state, done, hidden_states):
        self.buffer.append([state, action, reward, next_state, done, hidden_states])

    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done, hidden_states = zip(*sample)

        states = torch.stack([torch.tensor(s, dtype=torch.float32).to(device) for s in states])
        actions = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.stack([torch.tensor(s, dtype=torch.float32).to(device) for s in next_states])
        done = torch.tensor(done, dtype=torch.int64).to(device)

        hidden_states_h = torch.cat([h[0].view(1, 1, -1) for h in hidden_states], dim=1)
        hidden_states_c = torch.cat([h[1].view(1, 1, -1) for h in hidden_states], dim=1)

        hidden_states = (hidden_states_h, hidden_states_c)

        return states, actions, rewards, next_states, done, hidden_states

    def size(self):
        return len(self.buffer)

class DRQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DRQN, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.epsilon = args.eps

        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=self.hidden_dim, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, self.action_dim)

        self.model = self.create_model().to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=args.lr)

    def create_model(self):
        conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2)
        conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(400))  # width
        convh = conv2d_size_out(conv2d_size_out(600))  # height
        linear_input_size = convw * convh * 64
        linear = nn.Linear(linear_input_size, self.hidden_dim)

        model = nn.Sequential(
            conv1,
            nn.LeakyReLU(),
            conv2,
            nn.LeakyReLU(),
            nn.Flatten(),
            linear,
        )

        return model

    def forward(self, x, hidden):
        x = self.model(x)
        x = x.view(-1, 1, self.fc.in_features)
        x, hidden = self.lstm(x, hidden)
        x = self.fc(x)
        return x, hidden

    def get_action(self, state, hidden=None):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        with torch.no_grad():
            if hidden is None:
                hidden = (torch.zeros(1, 1, self.hidden_dim).to(device), torch.zeros(1, 1, self.hidden_dim).to(device))
            q_value, hidden = self.forward(state, hidden)
            q_value = q_value.cpu().detach().numpy()[0]
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1), hidden
        return np.argmax(q_value), hidden


    def update(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

class Agent:
    def __init__(self, env, sequence_length, batch_size):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DRQN(self.state_dim, self.action_dim).to(self.device)
        self.target_model = DRQN(self.state_dim, self.action_dim).to(self.device)
        self.target_update()

        self.buffer = ReplayBuffer()
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)
        self.ROWS = 600
        self.COLS = 400
        self.FRAME_STEP = 3
        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS))
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS)

        self.sequence_length = sequence_length
        self.batch_size = batch_size

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        states, actions, rewards, next_states, done, hidden_states = self.buffer.sample()

        hidden_h, hidden_c = hidden_states
        hidden = (hidden_h.view(1, args.batch_size, -1).contiguous(),
                  hidden_c.view(1, args.batch_size, -1).contiguous())

        q_values, _ = self.model(states, hidden)
        with torch.no_grad():
            next_hidden = (torch.zeros_like(hidden_h), torch.zeros_like(hidden_c))
            next_q_values, _ = self.target_model(next_states, next_hidden)
            next_q_values = next_q_values[0].max(dim=1).values

        # Obtain the Q-values corresponding to the selected actions
        q_values = q_values.squeeze().gather(1, actions.unsqueeze(1)).squeeze(1)

        # Calculate the target Q-values
        target_values = rewards + (1 - done.float()) * next_q_values.float() * args.gamma

        # Compute the loss
        loss = nn.MSELoss()(q_values, target_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    #힘들었다.
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
            hidden = None

            while not done:
                pygame.display.update()
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action, hidden = self.model.get_action(state, hidden)
                _, reward, done = self.step(action)
                epi_rewad += reward
                next_state = np.array(pygame.surfarray.array3d(screen))
                next_state = next_state.transpose((2, 0, 1))

                if done:
                    if ep % 10 == 0:
                        self.target_update()
                self.buffer.put(state.squeeze().cpu().numpy(), action, reward, next_state, done, hidden)
                total_reward += reward
                state = next_state

                i += 1

            if self.buffer.size() >= args.batch_size:
                self.replay()

            print('EP{} EpisodeReward={} total_step={}'.format(ep, total_reward, i))
            f = open("DRQN_Pic_epi_reward.txt", 'a')
            f.write(epi_rewad.__str__())
            f.write("\n")
            f.close()

            if epi_rewad >= 500:
                break

        end_time = time.time()
        f = open("DRQN_Pic_complet_time.txt", 'a')
        f.write((end_time - start_time).__str__())
        f.write("\n")
        f.close()

        # Save the model
        torch.save(self.model.state_dict(), "./DRQN_Pic_model.pt")
        self.env.close()
        pygame.quit()
        sys.exit()

def main():
    env = gym.make('CartPole-v1', new_step_api=True, render_mode='human').unwrapped
    agent = Agent(env, args.sequence_length, args.batch_size)
    agent.training(max_episodes=10000)

if __name__ == "__main__":
    main()

'''
코드의 변경된 부분은 다음과 같습니다:

ActionStateModel 클래스를 DRQN 클래스로 변경했습니다.
DRQN 클래스의 create_model 메서드에서 LSTM 레이어를 추가했습니다. 이렇게 하면 시간에 따른 종속성을 학습할 수 있습니다.
Agent 클래스에서 모델 생성 시 DRQN 클래스를 사용하도록 변경했습니다.

DQN과 DRQN의 차이점은 다음과 같습니다:

DQN은 딥 Q-네트워크(Deep Q-Network)로, 강화학습에서 Q-함수를 근사하는 데 사용되는 신경망입니다. 
DQN은 Convolutional Neural Network(CNN)와 Fully Connected(FC) 레이어를 사용하여 비디오 게임 화면과 같은 고차원 입력을 처리할 수 있습니다. 
이러한 신경망은 공간 정보를 처리하는 데 유용하지만, 시간에 따른 종속성을 처리하는 데는 제한적입니다.

DRQN은 DQN에 기반한 모델로, 시퀀스 데이터를 처리하는 데 더 효과적인 Recurrent Neural Network(RNN) 구성 요소를 포함합니다. 
DRQN은 기본 DQN 아키텍처에 LSTM 레이어를 추가하여 시간에 따른 종속성을 학습할 수 있습니다. 
이로 인해 DRQN은 시간적인 정보를 처리하는 데 더 강력한 모델이 됩니다.

따라서 주요 차이점은 DRQN이 시간에 따른 종속성을 처리하는 데 더 효과적인 방법을 사용한다는 것입니다. 
이는 주로 강화학습에서 효과적인 행동 결정에 도움이 됩니다.
'''