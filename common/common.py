import glob, io
import base64

from IPython import display as ipythondisplay
from pyvirtualdisplay import Display
from IPython.display import HTML

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

def random_episode(env):
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
    show_video('video_random_episode')
