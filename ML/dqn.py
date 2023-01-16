from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm

from common.common import show_video

class QNetwork(nn.Module):
    def __init__(self, inp_dim = 4, out_dim = 2):
        super().__init__()

        self.fc1 = nn.Linear(inp_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.head = nn.Linear(64, out_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

##### Expericence Replay #####
# State, Action, Reward, Next state, Done을 저장하고, 
# random sampling 할 수 있는 Experience Replay 를 구현  
class ReplayMemory:
    def __init__(self, buffer_size:int = 50000, n_steps =1, discount_rate=0.99):
        self.buffer = deque(maxlen = buffer_size)

    def write(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, sample_size:int=32):
        states, actions, rewards, next_states, dones = list(), list(), list(), list(), list()
        sample_indices = np.random.choice(len(self.buffer), size=sample_size, replace=False)

        for idx in sample_indices:
            sample = self.buffer[idx]
            states.append(sample[0])
            actions.append(sample[1])
            rewards.append(sample[2])
            next_states.append(sample[3])
            dones.append(sample[4])
        return states, actions, rewards, next_states, dones

def train(env_make, param_dict):
    replay_memory = ReplayMemory(buffer_size=param_dict['memory_size'],
                                 n_steps=param_dict['n_steps'],
                                 discount_rate=param_dict['discount_rate'])

    online_net = QNetwork()
    target_net = QNetwork()
    target_net.load_state_dict(online_net.state_dict())

    optimizer = torch.optim.Adam(online_net.parameters(),
                                lr = param_dict['learning_rate'])

    losses = []
    episode_reward = 0.0
    episode_rewards = []
    episode_steps = []

    print('Experience Replay Warmup...')
    state, done = env_make.reset(), False

    for warmup_step in tqdm.trange(param_dict['warmup_steps']):
        action = env_make.action_space.sample()
        next_state, reward, done, _ = env_make.step(action)

        replay_memory.write(state, action, reward / 100, next_state, done)

        if done:
            state = env_make.reset()
        else:
            state = next_state

    print('DQN Training Starts...')
    progress_bar = tqdm.trange(param_dict['train_steps'])

    state, done = env_make.reset(), False

    for step in progress_bar:
        if np.random.random() < param_dict['epsilon']:
            action = env_make.action_space.sample()
        else:
            tensor_state = torch.FloatTensor(state)
            action = online_net(tensor_state).argmax().item()

        next_state, reward, done, _ = env_make.step(action)

        replay_memory.write(state, action, reward/100, next_state, done)
        episode_reward += reward

        if done:
            state = env_make.reset()
            episode_rewards.append(episode_reward)
            episode_steps.append(step)
            episode_reward = 0.0
        else:
            state = next_state

        if step % param_dict['train_freq'] == 0:
            online_net.train()
            states, actions, rewards, next_states, dones = replay_memory.sample(param_dict['batch_size'])

            states = torch.FloatTensor(np.array(states))
            actions = torch.LongTensor(np.array(actions))
            rewards = torch.FloatTensor(np.array(rewards))
            next_states = torch.FloatTensor(np.array(next_states))
            dones = torch.FloatTensor(np.array(dones))

            with torch.no_grad():
                next_q_values = target_net(next_states).max(axis=-1)[0]
                target_q_values = rewards + param_dict['discount_rate'] * next_q_values * (1-dones)

            current_q_values = online_net(states).gather(-1, actions.unsqueeze(-1)).squeeze()

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if step % param_dict['target_update_freq'] == param_dict['target_update_freq'] - 1:
            target_net.load_state_dict(online_net.state_dict())

        if step % 100 == 99:
            avg_rew = np.mean(episode_rewards[-100:])
            progress_bar.set_description(f'loss: {np.mean(losses[-100:]):6.4f}, '
                                         f'epi. rew. : {avg_rew:5.1f}, '
                                         f'max_rew.: {np.max(episode_rewards):5.1f}')
            if avg_rew > param_dict['target_score']:
                break

    return online_net, losses, episode_steps, episode_rewards

def test(test_env, online_net, video_folder):

    rewards = 0.0

    ob, done = test_env.reset(), False
    while not done:
        action = online_net(torch.FloatTensor(ob)).argmax().item()
        ob, reward, done, info = test_env.step(action)
        rewards += reward
    test_env.close()
    print(f"\nEpisode Finished! Reward: {rewards:.1f}.  Please wait until GIF Video is created...")
    show_video(video_folder)
