
from gym.wrappers import Monitor


import gym

from time import sleep
import numpy as np
import random

from common.common import *
from ML import q_learning


# TODO error 생김
# display = Display(visible=0, size=(1400, 900))
# display.stat()

### random_episode ###

def learn_random_episode(env_make, video_folder):
    env = Monitor(env_make, f'./{video_folder}', force = True)
    random_episode(env, video_folder)

def _q_learning(env_make, train_steps, target_score, video_folder):
    env = Monitor(env_make, f'./{video_folder}', force = True)
    q_table = np.ones([env.observation_space.n, env.action_space.n])
    q_learning.train(env, parm_dict, q_table, train_steps = train_steps, target_score = target_score)
    score_viz = q_learning.test(env, q_table, video_folder, visualize=True)
    print(f'score : {score_viz}')
    reward_tot = 0
    for _ in range(100):
        reward_tot += q_learning.test(env, q_table, video_folder)

    print(f'reward_total : {reward_tot / 100}')
    print('visualize')
    visualize(q_table)

def visualize(q_table):
    dirs = ['◁','▽','▷','△']
    for row in range(4):
        txt = ''
        for col in range(4):
            txt += dirs[np.argmax(q_table[row*4 + col])]
        print(txt)
    print('\n')
    for row in range(4):
        txt = ''
        for col in range(4):
            # Printing the direction of maximum Q value
            txt = txt + " " + str(q_table[row * 4 + col])
        print(txt)

if __name__ == "__main__":
    print("##### FrozenLake-v1 no_slippery random_episode #####")
    env_make = gym.make("FrozenLake-v1", is_slippery=False)
    learn_random_episode(env_make,'FrozenLake-v1_no_slippery/random_episode')

    print("\n##### FrozenLake-v1 no_slippery Q_Learning #####")
    parm_dict = {'epsilon': 0.1,
    'alpha' : 0.1,
    'gamma' : 0.99}

    _q_learning(env_make, train_steps = 200000,
               target_score = 0.9,
               video_folder='FrozenLake-v1_no_slippery/Q_Learning')

    print("\n##### Q_Learning is_slippery=True #####")
    env_make = gym.make("FrozenLake-v1", is_slippery=True)

    _q_learning(env_make,
               train_steps = 80000000,
               target_score = 0.4,
               video_folder='FrozenLake-v1_slippery/Q_Learning')









