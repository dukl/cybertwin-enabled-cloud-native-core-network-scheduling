import numpy as np
import keras.backend as K

from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from keras.optimizers import RMSprop


class Critic:
    def __init__(self, inp_dim, lr):
        self.inp_dim, self.lr = inp_dim, lr
        self.discounted_r = K.placeholder(shape=(None,))
        self.model = self.build_model()
        self.rms_optimizer = RMSprop(lr=lr, epsilon=0.1, rho=0.99)

    def optimizer(self):
        critic_loss = K.mean(K.square(self.discounted_r - self.model.output))
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], critic_loss)
        return K.function([self.model.input, self.discounted_r], [], updates=updates)

    def save(self):
        self.model.save_weights('../../results/critic.h5')

    def load_weights(self):
        self.model.load_weights('../../results/critic.h5')


    def build_model(self):
        inp = Input(shape=(self.inp_dim,))
        x = Dense(128, activation='relu')(inp)
        x = Dense(128, activation='relu')(x)
        x = Dense(64, activation='relu')(x)
        out = Dense(1, activation='linear')(x)
        return Model(inp, out)
