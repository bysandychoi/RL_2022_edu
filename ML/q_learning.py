import random
import numpy as np

from IPython.display import clear_output

from common.common import show_video

def train(env, parm_dict, q_table, train_steps, target_score):
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
            if random.random() > parm_dict['epsilon']:
                action = np.argmax(q_table[state])
            else:
                action = env.action_space.sample()

            next_state, reward, done, info = env.step(action)
            reward_sum += reward

            q_table[state][action] = (1 - parm_dict['alpha']) * q_table[state][action] + parm_dict['alpha'] * (
                        reward + parm_dict['gamma'] * (1 - done) * np.max(q_table[next_state]))
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
