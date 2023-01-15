import glob, io
import base64
import numpy as np

from IPython import display as ipythondisplay
from matplotlib import pyplot as plt
from IPython.display import HTML, clear_output

def show_video(video_folder):
    mp4list =glob.glob(f'{video_folder}/*.mp4')
    if len(mp4list) > 0:
        mp4 = mp4list[0]
        video = io.open(mp4, 'r+b').read()
        encoded = base64.b64encode(video)
        ipythondisplay.display(HTML(data='''
            <video alt="test" autoplay loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
            </video>'''.format(encoded.decode('ascii'))))
    else:
        print("Could not find video")

def random_episode(env, video_folder):
    state = env.reset()
    done = False
    step_count = 0
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)
        state = next_state

        env.render(mode ='rgb_array')
        step_count += 1
        if step_count > 10:
            break
    env.close()
    show_video(video_folder)

def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w

def plot_losses(losses, title, xlabel, ylabel, save_path):
    plt.plot(moving_average(losses, 100))
    plot_common(title, xlabel, ylabel, save_path)

def plot_episode_rewards(episode_steps, episode_rewards, title, xlabel, ylabel, save_path):
    plt.plot(moving_average(episode_steps, 100), moving_average(episode_rewards, 100))
    plot_common(title, xlabel, ylabel, save_path)

def plot_common(title, xlabel, ylabel, save_path):
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
    plt.savefig(save_path)