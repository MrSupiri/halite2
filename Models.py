import random
import numpy as np
from collections import deque
import os
from contextlib import redirect_stdout
from shutil import copyfile
with open(os.devnull, 'w') as f:
    with redirect_stdout(f):
        import tensorflow as tf
        from keras.layers import Dense
        from keras.optimizers import Adam
        from keras.models import Sequential
        from keras.backend.tensorflow_backend import set_session
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        tf.logging.set_verbosity(tf.logging.ERROR)
        set_session(tf.Session(config=config))


class DQNAgent:
    def __init__(self, MODEL_NAME, state_size, action_size):
        self.model_loaded = False
        self.state_size = state_size
        self.action_size = action_size
        self.MODEL_NAME = MODEL_NAME + '.h5'
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

        if MODEL_NAME.startswith('TrainingBot') and os.path.isfile('models/ProjectFaker.h5'):
            copyfile('models/ProjectFaker.h5', 'models/{}'.format(self.MODEL_NAME ))

        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isfile('models/'+self.MODEL_NAME):
            self.model.load_weights('models/'+self.MODEL_NAME)
            self.model_loaded = True

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done=False):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            out = [0, 0, 0]
            out[random.randrange(self.action_size)] = 1
            return out
        act_values = self.model.predict(state)
        return act_values[0]  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self):
        self.model.save_weights('models/'+self.MODEL_NAME)
