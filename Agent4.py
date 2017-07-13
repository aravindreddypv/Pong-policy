import gym
import numpy as np

from keras import layers
from keras.models import Model
from keras import backend as K
from keras import utils as np_utils
from keras import optimizers
from gym import wrappers


class Agent(object):
    def __init__(self, input_dim, output_dim, hidden_dims):
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.__build_network(input_dim, output_dim, hidden_dims)
        self.__build_train_fn()

    def __build_network(self, input_dim, output_dim, hidden_dims):
        self.X = layers.Input(shape=(input_dim,))
        net = self.X

        for h_dim in hidden_dims:
            net = layers.Dense(h_dim)(net)
            net = layers.Activation("relu")(net)

        net = layers.Dense(output_dim)(net)
        net = layers.Activation("softmax")(net)

        self.model = Model(inputs=self.X, outputs=net)

    def __build_train_fn(self):
        action_prob_placeholder = self.model.output
        action_onehot_placeholder = K.placeholder(shape=(None, self.output_dim),
                                                  name="action_onehot")
        discount_reward_placeholder = K.placeholder(shape=(None,),
                                                    name="discount_reward")

        action_prob = K.sum(action_prob_placeholder * action_onehot_placeholder, axis=1)
        log_action_prob = K.log(action_prob)

        loss = - log_action_prob * discount_reward_placeholder
        loss = K.mean(loss)
        adam = optimizers.Adam(lr=1e-2)

        updates = adam.get_updates(params=self.model.trainable_weights,
                                   constraints=[],
                                   loss=loss)

        self.train_fn = K.function(inputs=[self.model.input,
                                           action_onehot_placeholder,
                                           discount_reward_placeholder],
                                   outputs=[],
                                   updates=updates)

    def get_action(self, state):
        action_prob = self.model.predict(state)
	# print action_prob
        action = np.random.choice(self.output_dim, p=np.reshape(action_prob, self.output_dim))
        return action

    def fit(self, S, A, R):
        action_onehot = np_utils.to_categorical(A, num_classes=self.output_dim)
        discount_reward = compute_discounted_R(R)

        assert S.shape[1] == self.input_dim, "{} != {}".format(S.shape[1], self.input_dim)
        assert action_onehot.shape[0] == S.shape[0], "{} != {}".format(action_onehot.shape[0], S.shape[0])
        assert action_onehot.shape[1] == self.output_dim, "{} != {}".format(action_onehot.shape[1], self.output_dim)
        assert len(discount_reward.shape) == 1, "{} != 1".format(len(discount_reward.shape))

        self.train_fn([S, action_onehot, discount_reward])


def compute_discounted_R(R, discount_rate=.99):
    discounted_r = np.zeros_like(R, dtype=np.float32)
    running_add = 0
    temp=0
    k=0
    for t in reversed(range(len(R))):
        running_add = running_add * discount_rate + R[t]
        discounted_r[t] = running_add
	if discounted_r[t]!=0 and k==0:
		temp=1
		k=1
    if temp!=0:	
    	discounted_r -= discounted_r.mean() / discounted_r.std()
    # print discounted_r

    return discounted_r


def prepro(I):
    I = I[35:195]  # crop
    I = I[::2, ::2, 0]  # downsample by factor of 2
    I[I == 144] = 0  # erase background (background type 1)
    I[I == 109] = 0  # erase background (background type 2)
    I[I != 0] = 1  # everything else (paddles, ball) just set to 1
    return I.astype(np.float).ravel()


def run_episode(env, agent):
    done = False
    S = []
    A = []
    R = []

    prev_x = None
    observation = env.reset()

    total_reward = 0

    while not done:
        # env.render()
        cur_x = prepro(observation)
        x = cur_x - prev_x if prev_x is not None else np.zeros(6400)
        prev_x = cur_x
        observ_array = np.array(x)  # convert to an array to give it to keras
        observ_array = observ_array.reshape(1, len(x))
        action = agent.get_action(observ_array)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        S.append(x)
        A.append(action)
        R.append(reward)

        if done:
            S = np.array(S)
            A = np.array(A)
            R = np.array(R)

            agent.fit(S, A, R)

    return total_reward


def main():
    env = gym.make("Pong-v0")
    env = wrappers.Monitor(env, 'Pong-experiment-2')
    input_dim = 6400
    output_dim = env.action_space.n
    agent = Agent(input_dim, output_dim, [64])
    t=0
    while True:
        reward = run_episode(env, agent)
        print reward,t
        if t % 100 == 0:
            agent.model.save("abcd.h5")
	t+=1
    env.close()
	

if __name__ == '__main__':
    main()
