import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os
import tensorflow as tf
from shutil import copytree

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '99'
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
tf.logging.set_verbosity(tf.logging.ERROR)


class SupiriNet(object):
    def __init__(self, MODEL_NAME, IN_SIZE, OUT_SIZE, LR=1e-3):
        self.LR = LR
        self.MODEL_NAME = MODEL_NAME
        self.INPUT_SIZE = IN_SIZE
        self.OUTPUT_SIZE = OUT_SIZE
        self.model_loaded = False

        self.X = tf.placeholder(tf.float32, [None, IN_SIZE])
        self.LOSS = tf.reduce_mean(tf.Variable(0, dtype=tf.float32))

        self.NN = self.NeuralNetwork(self.X)
        self.Optimizer = tf.train.AdamOptimizer(LR).minimize(self.LOSS)

        self.sess = tf.Session(config=config)
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

        if MODEL_NAME.startswith('TrainingBot') and os.path.isfile('model/ProjectFaker/ProjectFaker.meta'):
            copytree('model/ProjectFaker', 'model/{}'.format(MODEL_NAME))
            for file in os.listdir('model/ProjectFaker'):
                newname = str(file).replace('ProjectFaker', MODEL_NAME)
                os.rename('model/' + MODEL_NAME + '/' + file, 'model/' + MODEL_NAME + '/' + newname)

        self.model_dir = os.path.join('model', MODEL_NAME)
        self.model_dir = os.path.join(self.model_dir, MODEL_NAME)
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        if os.path.isfile(self.model_dir + '.meta'):
            self.saver.restore(self.sess, self.model_dir)
            self.model_loaded = True

    def NeuralNetwork(self, INPUT):
        network = tf.contrib.layers.fully_connected(INPUT, 64)
        network = tf.contrib.layers.fully_connected(network, 128)
        network = tf.contrib.layers.fully_connected(network, 128)
        network = tf.contrib.layers.fully_connected(network, 64)
        network = tf.contrib.layers.fully_connected(network, self.OUTPUT_SIZE)
        output = tf.nn.softmax(tf.reshape(network, [-1, self.OUTPUT_SIZE]))
        return output

    def learn(self, LOSS):
        self.sess.run(self.Optimizer, feed_dict={self.LOSS: LOSS})

    def predict(self, X):
        return self.sess.run(self.NN, feed_dict={self.X: X})[0]

    def save(self):
        self.saver.save(self.sess, self.model_dir)

    def get_loss(self):
        return self.sess.run(self.LOSS)

    def update_loss(self, LOSS):
        self.sess.run(self.LOSS, feed_dict={self.LOSS: LOSS})


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

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

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
