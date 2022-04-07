from tensorflow import keras
import numpy as np
import gym

model = keras.models.load_model('./save/cartpole.h5')
env = gym.make("CartPole-v1")
total_score = 0
for i in range(10):
    s = env.reset()
    score = 0
    while True:
        env.render()
        a = np.argmax(model.predict(s.reshape(1, 4)))
        next_s, reward, done, info = env.step(a)
        score += reward
        s = next_s
        if done:
            total_score += score
            print('episode {} score = {}'.format(i+1, score))
            break
print('average score = {}'.format(total_score / 10))
