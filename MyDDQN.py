from collections import deque
import random

import gym

from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt


class DDQN:
    def __init__(self, state_size, action_size
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = 0.001
        self.reward_decay = 0.95    # discount rate
        self.replay_memory = deque(maxlen=2000)
        self.batch_size = 32
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.model = self.create_q_model()
        self.target_model = self.create_q_model()
        self.update_model_target()
        self.step = 0
        self.Q_value = []

    def create_q_model(self):
        model = keras.Sequential([
            layers.Dense(32, input_dim=self.state_size, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(self.action_size, activation='linear')
        ])
        # sgd = optimizers.SGD(learning_rate=self.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
        adam = optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(loss='mse', optimizer=adam)

        # huber loss
        # model.compile(loss=keras.losses.Huber(), optimizers=adam)
        return model

    def choose_action(self, state):
        actions_value = self.model.predict(state)
        if np.random.uniform() <= self.epsilon:  # 采用随机动作
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
            action = np.random.randint(self.action_size)
        else:
            action = np.argmax(actions_value[0])
        self.Q_value.append(actions_value[0][action])
        return action

    def learn(self):
        batch = random.sample(self.replay_memory, self.batch_size)
        for state, action, reward, next_state, done in batch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                # DQN
                # target[0][action] = reward + self.gamma * np.amax(t)

                # DDQN
                a = self.model.predict(next_state)[0]
                target[0][action] = reward + self.reward_decay * t[np.argmax(a)]

            # 训练决策网络
            self.model.fit(state, target, epochs=1, verbose=0)

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
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    rewards = []
    for episode in range(1000):
        state = env.reset()
        episode_reward = 0
        while True:
            # env.render()
            RL.step += 1
            state = np.reshape(state, [1, state_size])
            action = RL.choose_action(state)
            next_state, reward, done, info = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            RL.store_transition(state, action, reward, next_state, done)
            episode_reward += reward
            state = next_state
            if len(RL.replay_memory) > 32:
                RL.learn()
            if done:
                RL.update_model_target()
                rewards.append(episode_reward)
                print("{} Episode, score = {} max = {} e = {}".
                      format(episode + 1, episode_reward, np.max(rewards), RL.epsilon))
                break
        # 提前终止训练
        if len(rewards) >= 6 and np.mean(rewards[-6:]) >= 180:
            break
    RL.save('./save/')
    file = open('./save/rewards.txt', 'w')
    file.write(str(rewards))
    file.close()
    # plt.plot(np.array(rewards), c='r')
    # plt.show()


def test():
    model = keras.models.load_model('./save/' + name + 'cartpole.h5')
    score = 0
    for i in range(10):
        s = env.reset()
        while True:
            # env.render()
            a = np.argmax(model.predict(s.reshape(1, 1, 4)))
            next_s, reward, done, info = env.step(a)
            score += reward
            s = next_s
            if done:
                break
    print('average score = {}'.format(score / 10))


if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    name = 'test_1_'
    RL = DDQN(env.observation_space.shape[0], env.action_space.n)
    train()
    # test()
