import gym
from gym.wrappers import Monitor
from gym.envs.classic_control import CartPoleEnv
from gym.spaces import Discrete
from gym import register

from ML import dqn, q_learning
from common.common import *

def learn_random_episode(env_make, video_folder, need_print = False):
    env = Monitor(env_make, f'./{video_folder}', force = True)

    if need_print == True:
        print(f'observation_space : {env.observation_space}')
        print(f'action_space : {env.action_space}')

    random_episode(env, video_folder)

def _q_learning(train_env_make, test_env_make, param_dict, video_folder):
    train_env = Monitor(train_env_make, f'./{video_folder}', force = True)
    q_table = np.ones([train_env.observation_space.n, train_env.action_space.n])
    q_learning.train(train_env, param_dict, q_table,
                     train_steps = param_dict['train_steps'], target_score = param_dict['target_score'])

    test_env = Monitor(test_env_make, f'./{video_folder}', force = True)
    q_learning.test(test_env, q_table, video_folder, visualize = True)

def _dqn(train_env_make, test_env_make, video_folder, param_dict):
    online_net = dqn.train(train_env_make, param_dict)
    test_env = Monitor(test_env_make, f'./{video_folder}', force=True)
    dqn.test(test_env, online_net, video_folder)

class DiscreteCartpole(CartPoleEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bin_size = 7
        self.observation_space = Discrete(self.bin_size**4)
    def discretize_state(self, state):
        discrete_state = 0
        state[0] *= 100
        state[2] *= 10
        for value in state:
            d_v = int((value+1)*self.bin_size/2)
            if d_v < 0:
                d_v = 0
            elif d_v >= self.bin_size:
                d_v = self.bin_size -1
            discrete_state *= self.bin_size
            discrete_state += d_v
        return discrete_state

    def reset(self):
        state = super().reset()
        return self.discretize_state(state)

    def step(self, action):
        state, reward, done, info = super().step(action)
        return self.discretize_state(state), reward, done, info

if __name__ == "__main__":
    print("##### CartPole-v1 random_episode #####")
    env_make = gym.make('CartPole-v1')
    learn_random_episode(env_make, 'CartPole-v1/random_episode')

    register(
        id = 'DiscreteCartpole-v0',
        entry_point = '__main__:DiscreteCartpole',
        max_episode_steps = 200
    )

    print("\n##### DiscreteCartpole-v0 random_episode #####")
    env_make = gym.make('DiscreteCartpole-v0')
    learn_random_episode(env_make, 'DiscreteCartpole-v0', need_print = True)

    print("\n##### DiscreteCartpole-v0 Q_Learning #####")
    train_env_make = gym.make('DiscreteCartpole-v0')
    test_env_make = gym.make('DiscreteCartpole-v0')

    param_dict_q_learing = {'epsilon': 0.1,
                            'alpha' : 0.1,
                            'gamma' : 0.99,
                            'train_steps' : 5000000,
                            'target_score' : 150}
    _q_learning(train_env_make, test_env_make, param_dict=param_dict_q_learing, video_folder = 'DiscreteCartpole-v0')

    print("\n##### CartPole-v0 DQN #####")
    train_env_make = gym.make("CartPole-v0")
    test_env_make = gym.make('CartPole-v0')

    # hyperparameter
    param_dict_DQN = {'target_update_freq' : 512, # 128 ~ 4096
                    'warmup_steps' : 10000, # batch_size ~ memory_size
                    'train_steps' : 100000,
                    'train_freq' : 4, # 1 ~ 4
                    'memory_size' : 100000, # 10000 ~ 1000000

                    'epsilon' : 0.01, # 0.001 ~ 0.1

                    'discount_rate' : 0.99, # 0.9 ~ 0.9999

                    'batch_size' : 64, # 16 ~ 256
                    'learning_rate' : 0.0001, # 0.00001~ 0.001

                    'n_steps' : 8,

                    'target_score' : 195}

    _dqn(train_env_make, test_env_make, 'CartPole-v0/DQN', param_dict_DQN)








