"""
CNN-DQN based RL agent for Pong

Requirements:
- pytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import math
import random
import os
import sys
from collections import namedtuple

class Agent:

    NAME = 'AgentPongai'
    MODEL_PATH = 'models/'
    TRAINED_MODEL = 'model.mdl'
    N_ACTIONS = 3
    FRAME_WIDTH = 200
    FRAME_HEIGHT = 200
    EPS_START = 0.99
    EPS_END = 0.05
    EPS_DECAY = 150000
    LR = 1e-4

    # Common interface for both training and testing
    # supports 0 (frames are summed to each other rather than stacked), 2 of 4 frame stacking

    def __init__(   self, training=False, batch_size=64, gamma=0.99, memory_size=15000, \
                    stack_frames=4, downscale=True, priority_memory=True):
        self.training = training
        self.batch_size = batch_size
        self.gamma = gamma
        self.stack_frames = stack_frames if stack_frames in (0, 2, 4) else 0
        self.downscale = downscale
        self.priority_memory = priority_memory
        self.w = Agent.FRAME_WIDTH // 2 if downscale else Agent.FRAME_WIDTH
        self.h = Agent.FRAME_HEIGHT // 2 if downscale else Agent.FRAME_HEIGHT
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        in_channels = stack_frames if stack_frames in (2, 4) else 1
        self.policy_net = DQN(Agent.N_ACTIONS, in_channels, self.h, self.w).to(self.device)
        self.target_net = DQN(Agent.N_ACTIONS, in_channels, self.h, self.w).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.memory = PriorityMemory(memory_size) if priority_memory else ReplayMemory(memory_size)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Agent.LR)
        self.epsilon = 1.0
        self.prev_state = [None] if self.stack_frames in (0, 2) else [None, None, None]
        self.steps_done = 0

    """
    Load the trained model for evaluation
    """
    def load_model(self):
        self.policy_net.load_state_dict(torch.load(Agent.TRAINED_MODEL, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.policy_net.eval()

    """
    This needs to correspond to save
    """
    def load_trained_model(self):
        dir = os.path.join(os.getcwd(), Agent.MODEL_PATH, self.get_name())
        if not os.path.isdir(dir):
            sys.exit(f'Model loading failed: {dir} does not exist')
        model_files = os.listdir(dir)
        model_files.sort()
        model_path = os.path.join(dir, model_files[-1]) # Select the latest model
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Agent.LR)

    """
    Load a trained model from a specific path
    """
    def load_model_file(self, model_path):
        self.policy_net.load_state_dict(torch.load(model_path, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=Agent.LR)

    """
    Reset Agent when an episode has finished
    """
    def reset(self):
        self.prev_state = [None] if self.stack_frames in (0, 2) else [None, None, None]

    """
    frame: numpy array (200, 200, 3) -> convert to pytorch tensor
    """
    def get_action(self, frame=None):
        self.epsilon = Agent.EPS_END + (Agent.EPS_START - Agent.EPS_END) * math.exp(-1. * self.steps_done / Agent.EPS_DECAY)
        self.steps_done += 1
        action = torch.randint(0, Agent.N_ACTIONS, (1,)).item()
        if frame is None:
            return action
        state = self.__preprocess_frame(frame)
        if self.prev_state[0] != None and (not self.training or np.random.uniform(size=1)[0] > self.epsilon):
            # Select the 'best' action
            with torch.no_grad():
                state_diff = self.__process_states(state, self.prev_state)
                action = torch.argmax(self.policy_net(state_diff)).item()
        if len(self.prev_state) == 3:
            # 4 frames stacking enabled
            self.prev_state[0] = self.prev_state[1]
            self.prev_state[1] = self.prev_state[2]
        self.prev_state[-1] = state
        return action

    """
    Get Agent name, used by the environment
    """
    def get_name(self):
        return Agent.NAME

    """
    Save a Pytorch model to MODEL_PATH
    """
    def save(self):
        dir = os.path.join(os.getcwd(), Agent.MODEL_PATH, self.get_name())
        if os.path.isdir(dir):
            # This does not work for more than 100 models
            content = os.listdir(dir)
            content.sort()
            model_id = content[-1][-6:-4] if len(content) > 0 else -1 # Get the latest model number
            model_id = '%02d' % (int(model_id) + 1, ) # Increase it by one
        else:
            os.makedirs(dir)
            model_id = '00'
        model_path = os.path.join(dir, self.get_name() + model_id + '.pth')
        torch.save(self.policy_net.state_dict(), model_path)

    """
    Convert a rbg image frame given as a numpy array to a grayscale tensor
    frame: np.array with shape(200, 200, 3) containing int values 0 - 255
    """
    def __preprocess_frame(self, frame):
        frame = frame.astype(float)
        frame = np.true_divide(frame, 255.0)
        frame = 1.0 - frame # invert colors so that the background is zeros
        grayscale = 0.299 * frame[:,:,0] + 0.587 * frame[:,:,1] + 0.114 * frame[:,:,2]
        if self.downscale:
            grayscale = grayscale.reshape((1, Agent.FRAME_HEIGHT//2, 2, Agent.FRAME_WIDTH//2, 2)).max(4).max(2) # downscale by factor of 2
        return torch.from_numpy(grayscale).to(torch.float32).to(self.device)

    """
    Combine two consequtive states to create a combined state that shows movement.
    This is the state information that is feed on the DQN and stored.
    state: torch float32 tensor with values in the range of 0.0 and 1.0
    prev_state: torch float32 tensor with values in the range of 0.0 and 1.0
    """
    def __state_diff(self, state, prev_state):
        combined = torch.add(state, 0.5 * prev_state).view(1, 1, self.h, self.w) # multiply prev_state to distinguish differences
        combined -= torch.min(combined)
        combined /= torch.max(combined)
        return combined

    """
    Stack two consequtive states to create a combined state that shows movement.
    This is the state information that is feed on the DQN and stored.
    states: torch float32 tensor with values in the range of 0.0 and 1.0 as a list
    """
    def __stack_states(self, states):
        stacked = torch.stack(states).view(1, self.stack_frames, self.h, self.w)
        return stacked

    """
    This is just a wrapper for processing state and the previous state
    """
    def __process_states(self, state, prev_states):
        return self.__stack_states([state] + prev_states) if self.stack_frames > 0 else self.__state_diff(state, prev_states[0])


    # Training interface

    """
    Store actions, states, rewards for replay
    Note: state and next_state need to be image frame differences as tensors
    observations: 3 latest observations (frames)
    """
    def update_memory(self, observations, action, reward, done):
        action = torch.tensor([[action]], dtype=torch.long, device=self.device)
        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)
        states = [self.__preprocess_frame(observation) for observation in observations]
        state = self.__process_states(states[-2], states[:-2])
        next_state = self.__process_states(states[-1], states[1:-1])
        done = torch.tensor([done], dtype=torch.bool, device=self.device)
        self.memory.push(state, action, next_state, reward, done)

    """
    Do a single batch update on the DQN network
    """
    def update_dqn(self):
        if len(self.memory) < self.batch_size:
            return
        if self.priority_memory:
            samples, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            samples = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*samples))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        non_final_mask = [done_batch == False]
        non_final_next_states = next_state_batch[non_final_mask]

        # Compute Q(s, a) by selecting Q values matching selected actions
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        # Compute V(s_t+1)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute target Q values
        target_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute loss and optimize the network
        loss = F.smooth_l1_loss(state_action_values, target_state_action_values, reduction='none')
        if self.priority_memory:
            priorities = torch.abs(loss).detach() + 1e-5
            loss = loss * weights
            self.memory.update_priorities(indices, priorities.cpu().numpy())

        self.optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-10, 10) # to make stable updates
        self.optimizer.step()

    """
    Double DQN: https://arxiv.org/pdf/1509.06461.pdf
    """
    def update_ddqn(self):
        if len(self.memory) < self.batch_size:
            return
        if self.priority_memory:
            samples, indices, weights = self.memory.sample(self.batch_size)
            weights = torch.tensor(weights, dtype=torch.float32, device=self.device)
        else:
            samples = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*samples))
        state_batch = torch.cat(batch.state)
        next_state_batch = torch.cat(batch.next_state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        done_batch = torch.cat(batch.done)
        non_final_mask = [done_batch == False]
        non_final_next_states = next_state_batch[non_final_mask]

        # Compute Q(s, a) by selecting Q values matching selected actions
        state_action_values = self.policy_net(state_batch).gather(1, action_batch).squeeze(1)
        # Compute V(s_t+1)
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        greedy_actions = self.policy_net(non_final_next_states).max(1)[1].unsqueeze(1).detach()
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).gather(1, greedy_actions).squeeze(1).detach()

        # Compute target Q values
        target_state_action_values = reward_batch + self.gamma * next_state_values

        # Compute loss and optimize the network
        loss = F.smooth_l1_loss(state_action_values, target_state_action_values, reduction='none')
        if self.priority_memory:
            priorities = torch.abs(loss).detach() + 1e-5
            loss = loss * weights
            self.memory.update_priorities(indices, priorities.cpu().numpy())

        self.optimizer.zero_grad()
        loss = loss.mean()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-10, 10) # to make stable updates
        self.optimizer.step()

    """
    Update the target network to match the policy network
    Call this periodically from the training loop
    """
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """
    Set agent learning rate
    """
    def set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr


"""
CNN-based neural network which is updated according to DQN update rules
"""
class DQN(nn.Module):

    def __init__(self, n_actions, in_channels, h=Agent.FRAME_HEIGHT, w=Agent.FRAME_WIDTH):
        super(DQN, self).__init__()
        # Conv2d: https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d
        padding, dilation = 0, 1
        strides = [4, 2, 1]
        kernels = [(8, 8), (4, 4), (3, 3)]
        channels = [in_channels, 32, 64, 64]
        def out_size(in_size, kernel_size, stride):
            return math.floor((in_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1)

        self.conv1 = nn.Conv2d(channels[0], channels[1], kernels[0], padding=padding, dilation=dilation, stride=strides[0])
        self.conv2 = nn.Conv2d(channels[1], channels[2], kernels[1], padding=padding, dilation=dilation, stride=strides[1])
        self.conv3 = nn.Conv2d(channels[2], channels[3], kernels[2], padding=padding, dilation=dilation, stride=strides[2])
        h_out = out_size(out_size(out_size(h, kernels[0][0], strides[0]), kernels[1][0], strides[1]), kernels[2][0], strides[2])
        w_out = out_size(out_size(out_size(w, kernels[0][1], strides[0]), kernels[1][1], strides[1]), kernels[2][1], strides[2])
        flattened_size = h_out * w_out * channels[-1]
        self.fc1 = nn.Linear(flattened_size, 512)
        self.fc2 = nn.Linear(512, n_actions)

    """
    x : (batch_size, in_channels, image_width, image_height) = a batch of  in_channels images as a tensor
    """
    def forward(self, x):
        y = F.relu(self.conv1(x))
        y = F.relu(self.conv2(y))
        y = F.relu(self.conv3(y))
        y = torch.flatten(y, start_dim=1)
        y = F.relu(self.fc1(y))
        y = self.fc2(y)
        return y


"""
Directly from Pytorch example implementation for replay memory
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
Transition = namedtuple('Transition',
                    ('state', 'action', 'next_state', 'reward', 'done'))
class ReplayMemory():
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def update_beta(self):
        pass

    def __len__(self):
        return len(self.memory)

"""
https://arxiv.org/pdf/1511.05952.pdf
"""
class PriorityMemory():
    def __init__(self, capacity, alpha=0.6, beta0=0.4):
        self.capacity = capacity
        self.memory = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.alpha = alpha
        self.beta0 = beta0
        self.beta = beta0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # store transition with maximal priority
        self.memory[self.position] = Transition(*args)
        self.priorities[self.position] = 1.0 if len(self.memory) == 1 else np.max(self.priorities)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        N = len(self.memory)
        priorities = self.priorities[:N]
        P = np.power(priorities, self.alpha)
        P /= np.sum(P)

        indices = np.random.choice(N, size=batch_size, p=P, replace=False)
        samples = [self.memory[i] for i in indices]

        w = np.power(N * P[indices], -self.beta)
        w /= np.max(w)
        return samples, indices, w

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority

    def update_beta(self, interval=8000):
        self.beta = min(1.0, self.beta + (1.0 - self.beta0) / interval)

    def __len__(self):
        return len(self.memory)



## Testing ##

def test_DQN():
    n_actions = 3
    dqn = DQN(n_actions, 1, h=200, w=200)
    test = torch.ones((1, 1, 200, 200))
    assert dqn(test).size() == torch.Size([1, n_actions])
    test2 = torch.zeros((13, 1, 200, 200))
    assert dqn(test2).size() == torch.Size([13, n_actions])

def test_Agent():
    agent = Agent()
    frame1 = np.ones((200, 200, 3))
    frame2 = np.ones((200, 200, 3))
    assert type(agent.get_action(frame1)) == int
    assert agent.get_action(frame2) in [i for i in range(Agent.N_ACTIONS)]

if __name__ == '__main__':
    test_DQN()
    test_Agent()
