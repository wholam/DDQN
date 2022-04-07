from collections import deque
import random

import gym

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt
from keras import backend as K

class DDQN:
    def __init__(self, state_size, action_size
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.0025
        self.reward_decay = 0.95    # discount rate
        self.replay_memory = deque(maxlen=20000)
        self.batch_size = 128
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self.create_q_model()
        self.target_model = self.create_q_model()
        self.update_model_target()
        self.step = 0
        self.Q_value = []

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def create_q_model(self):
        model = keras.Sequential([
            layers.Dense(32, input_dim=self.state_size, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        adam = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)
        # model.compile(loss=self._huber_loss, optimizer=adam)
        return model

    def choose_action(self, state):
        state = np.reshape(state, [1, self.state_size])
        actions_value = self.model.predict(state)
        if np.random.uniform() <= self.epsilon:  # 采用随机动作
            action = np.random.randint(self.action_size)
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            action = np.argmax(actions_value[0])
        self.Q_value.append(actions_value[0][action])
        return action

    def learn(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        state = np.array([i[0] for i in batch])
        action = np.array([i[1] for i in batch])
        reward = np.array([i[2] for i in batch])
        next_state = np.array([i[3] for i in batch])
        done = np.array([i[4] for i in batch])

        q = self.model.predict(state)
        next_q = self.model.predict(next_state)
        q_target = self.target_model.predict(next_state)

        for i in range(self.batch_size):
            if done[i]:
                q[i][action[i]] = reward[i]
            else:
                # DQN
                # q[i][0][action[i]] = reward[i] + self.reward_decay * np.amax(q_target[i][0])

                # DDQN
                q[i][action[i]] = reward[i] + self.reward_decay * q_target[i][np.argmax(next_q[i])]

        # 训练决策网络
        self.model.fit(state, q, epochs=1, verbose=0, batch_size=self.batch_size)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_memory.append((state, action, reward, next_state, done))

    def update_model_target(self):
        self.target_model.set_weights(self.model.get_weights())

    def save(self, path):
        # 要保存的值
        # model、Q值
        self.target_model.save(path + 'cartpole.h5')
        file = open(path + 'Q_value.txt', 'w')
        file.write(str(self.Q_value))
        file.close()


def train():
    path = './save/'
    rewards = []
    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        while True:
            # env.render()
            RL.step += 1
            action = RL.choose_action(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            RL.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if len(RL.replay_memory) > 128:
                RL.learn()
            if done:
                RL.update_model_target()
                rewards.append(episode_reward)
                print("{} Episode, score = {} max = {} e = {}".
                      format(episode + 1, episode_reward, np.max(rewards), RL.epsilon))
                break
        if episode % 10 == 0:
            RL.save(path)
            file = open(path + 'rewards.txt', 'w')
            file.write(str(rewards))
            file.close()
        # 提前终止训练
        # if len(rewards) >= 6 and np.mean(rewards[-6:]) >= 180:
        #     break
    RL.save(path)
    file = open(path + 'rewards.txt', 'w')
    file.write(str(rewards))
    file.close()
    # plt.plot(np.array(rewards), c='r')
    # plt.show()


def test():
    model = keras.models.load_model('./save/' + 'cartpole.h5')
    score = 0
    for i in range(10):
        s = env.reset()
        while True:
            env.render()
            a = np.argmax(model.predict(s.reshape(1, 4)))
            next_s, reward, done, info = env.step(a)
            score += reward
            s = next_s
            if done:
                break
    print('average score = {}'.format(score / 10))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    RL = DDQN(env.observation_space.shape[0], env.action_space.n)
    # train()
    test()
