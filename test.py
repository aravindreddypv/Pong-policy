import gym
import numpy as np

from keras import layers
from keras.models import Model
from keras import backend as K
from keras.models import load_model
from keras import utils as np_utils
from keras import optimizers
from gym import wrappers


class Agent(object):
    def __init__(self, input_dim, output_dim, model):
        self.input_dim = input_dim
        self.output_dim = output_dim
	self.model=model

    def get_action(self, state):
        action_prob = self.model.predict(state)
	print action_prob
        action = np.random.choice(self.output_dim, p=np.reshape(action_prob, self.output_dim))
        return action


def prepro(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def run_episode(env, agent):
    done = False

    prev_x = None
    observation = env.reset()

    total_reward = 0

    while not done:
        env.render()
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(6400)
        prev_x = cur_x
        observ_array = np.array(x)  # convert to an array to give it to keras
        observ_array = observ_array.reshape(1, len(x))
        action = agent.get_action(observ_array)
        observation, reward, done, info = env.step(action)
        total_reward += reward
    return total_reward


def main():
    env = gym.make("Pong-v0")
    # env = wrappers.Monitor(env, 'Pong-experiment-2')
    input_dim = 6400
    output_dim = env.action_space.n
    model=load_model('abcd2.h5')
    agent = Agent(input_dim,output_dim,model)
    while True:
        reward = run_episode(env, agent)
        print reward
    env.close()
	

if __name__ == '__main__':
    main()
