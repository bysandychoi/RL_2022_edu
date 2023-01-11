
from gym.wrappers import Monitor
from IPython.display import clear_output

import gym

from time import sleep
import numpy as np
import random

from common.common import *

# TODO error 생김
# display = Display(visible=0, size=(1400, 900))
# display.stat()

### random_episode ###

def learn_random_episode(env_make):
    env = Monitor(env_make, './video_random_episode', force = True)
    random_episode(env)

def q_learning(env_make, train_steps, target_score, video_folder):
    env = Monitor(env_make, f'./{video_folder}', force = True)
    q_table = np.ones([env.observation_space.n, env.action_space.n])
    train(env, q_table, train_steps = train_steps, target_score = target_score)
    score_viz = test(env, q_table, video_folder, visualize=True)
    print(f'score : {score_viz}')
    reward_tot = 0
    for _ in range(100):
        reward_tot += test(env, q_table, video_folder)

    print(f'reward_total : {reward_tot / 100}')
    print('visualize')
    visualize(q_table)

def train(env, q_table, train_steps, target_score):
    episode_num = 0
    reward_sum = 0
    total_steps = 0

    while total_steps < train_steps:
        episode_num += 1
        if episode_num % 100 == 0:
            clear_output(wait = True)
            print(f'{episode_num} ep : {reward_sum / 100}')
            if target_score <= reward_sum / 100:
                break
            reward_sum  = 0

        state = env.reset()
        done = False

        while not done:
            total_steps += 1
            if random.random() > epsilon:
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()

            next_state, reward, done, info = env.step(action)
            reward_sum += reward

            q_table[state][action] = (1 - alpha) * q_table[state][action] + alpha * (
                        reward + gamma * (1 - done) * np.max(q_table[next_state]))
            state = next_state

    print(f'Total stpes: {total_steps}')

def test(env, q_table, video_folder, visualize = False, step_limit = 500):
    state = env.reset()
    reward_sum = 0
    step_count = 0
    done = False
    while not done:
        step_count += 1
        if step_count > step_limit:
            break

        action = np.argmax(q_table[state])

        state, reward, done, info = env.step(action)

        reward_sum += reward
        if visualize:
            env.render(mode='rgb_array')

    env.close()
    if visualize:
        show_video(video_folder)

    return reward_sum

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

    env_make = gym.make("FrozenLake-v1", is_slippery=False)

    print("##### random_episode #####")
    learn_random_episode(env_make)

    print("\n##### Q_Learning is_slippery=False #####")
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.99

    q_learning(env_make, train_steps = 200000,
               target_score = 0.9,
               video_folder='video_q_learning_no_slippery')

    print("\n##### Q_Learning is_slippery=True #####")
    env_make = gym.make("FrozenLake-v1", is_slippery=True)

    q_learning(env_make,
               train_steps = 80000000,
               target_score = 0.9,
               video_folder='video_q_learning_slippery')









