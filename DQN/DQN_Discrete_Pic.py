import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import gym
import argparse
import numpy as np
from collections import deque
import random
import time
import cv2
import pygame
import sys

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
        #states = torch.stack([torch.tensor(x, dtype=torch.float32) for x in states])
        actions = torch.tensor(actions, dtype=torch.int64)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        #next_states = torch.stack([torch.tensor(x, dtype=torch.float32) for x in next_states])
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
    '''
    def create_model(self):
        model = nn.Sequential(
            nn.Linear(self.state_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.action_dim)
        )
        return model
    '''

    def create_model(self):
        conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2)
        conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2)
        conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)

        def conv2d_size_out(size, kernel_size=3, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(600)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(400)))
        linear_input_size = convw * convh * 32
        fc1 = nn.Linear(linear_input_size, 512)
        fc2 = nn.Linear(512, 256)
        fc3 = nn.Linear(256, self.action_dim)

        model = nn.Sequential(
            conv1,
            nn.ReLU(),
            conv2,
            nn.ReLU(),
            conv3,
            nn.ReLU(),
            nn.Flatten(),
            fc1,
            nn.ReLU(),
            fc2,
            nn.ReLU(),
            fc3
        )
        return model

    def forward(self, state):
        return self.model(state)
    '''
    def get_action(self, state):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        with torch.no_grad():
            state = torch.tensor(state).float().unsqueeze(0)
            q_value = self.forward(state).detach().numpy()[0]
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        return np.argmax(q_value)
    '''

    def get_action(self, state):
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float32)
            #state = state.reshape(1, 4, 600, 400)
            state = state.unsqueeze(0)
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
        self.ROWS = 600
        self.COLS = 400
        self.FRAME_STEP = 3
        self.image_memory = np.zeros((self.FRAME_STEP, self.ROWS, self.COLS))
        self.state_size = (self.FRAME_STEP, self.ROWS, self.COLS)

    def target_update(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self):
        #for _ in range(10):
        states, actions, rewards, next_states, done = self.buffer.sample()
        #states = states.reshape(32, 3, 600, 400)
        #next_states = next_states.reshape(32, 3, 600, 400)
        #states = torch.tensor(states, dtype=torch.float32)
        #next_states = torch.tensor(next_states, dtype=torch.float32)
        targets = self.target_model(states)
        with torch.no_grad():
            next_q_values = self.target_model(next_states).max(dim=1).values
        targets[range(args.batch_size), actions] = rewards + (1 - done) * next_q_values * args.gamma
        loss = nn.MSELoss()(targets, self.model(states))
        self.model.update(loss)
    '''
    def imshow(self, image, frame_step=0):
        cv2.imshow("cartpole" + str(frame_step), image[frame_step, ...])
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return

    def GetImage(self):
        img = self.env.render(mode='rgb_array')

        img_rgb = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_rgb_resized = cv2.resize(img_rgb, (self.COLS, self.ROWS), interpolation=cv2.INTER_CUBIC)
        img_rgb_resized[img_rgb_resized < 255] = 0
        img_rgb_resized = img_rgb_resized / 255

        self.image_memory = np.roll(self.image_memory, 1, axis=0)
        self.image_memory[0, :, :] = img_rgb_resized

        # self.imshow(self.image_memory,0)
        return np.expand_dims(self.image_memory, axis=0)
    def reset(self):
        self.env.reset()
        for i in range(self.FRAME_STEP):
            state = self.GetImage()
        return state
    '''
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
            screen = pygame.display.set_mode((600, 400))
            done, total_reward = False, 0
            state = self.env.reset()
            state = np.array(pygame.surfarray.array3d(screen))
            state = state.transpose((2, 0, 1)) / 255.0
            # state = state.astype(np.float32).reshape(-1)
            #state_tensor = torch.tensor(state, dtype=torch.float32)  # (3, 600, 400)
            epi_rewad = 0
            while not done:
                pygame.display.update()
                action = self.model.get_action(state)
                _, reward, done = self.step(action)

                next_state = np.array(pygame.surfarray.array3d(screen))
                next_state = next_state.transpose((2, 0, 1)) / 255.0
                #next_state = next_state.astype(np.float32).reshape(-1)
                #next_state = torch.tensor(next_state, dtype=torch.float32)

                epi_rewad += reward
                self.buffer.put(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state
                if self.buffer.size() >= args.batch_size:
                    self.replay()
                pygame.time.delay(250)#1000 -> 1second
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            f = open("DQN_epi_reward.txt", 'a')
            f.write(epi_rewad.__str__())
            f.write("\n")
            f.close()
        end_time = time.time()
        f = open("DQN_complet_time.txt", 'a')
        f.write((end_time - start_time).__str__())
        f.write("\n")
        f.close()
        # Close the environment and pygame window
        self.env.close()
        pygame.quit()
        sys.exit()

def main():
    env = gym.make('CartPole-v1', new_step_api=True, render_mode='human').unwrapped
    agent = Agent(env)
    agent.training(max_episodes=1000)

if __name__ == "__main__":
    main()