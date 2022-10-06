import numpy as np
import keras.backend as K

from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
from keras.initializers import RandomUniform
from keras.optimizers import RMSprop

class Actor:
    def __init__(self, id, inp_dim, out_dim, is_discret, lr, layer):
        self.inp_dim, self.out_dim, self.is_discret, self.lr, self.layer, self.id = inp_dim, out_dim, is_discret, lr, layer, id
        self.model = self.build_model()
        self.action_pl = K.placeholder(shape=(None, self.out_dim))
        self.advantages_pl = K.placeholder(shape=(None,))
        self.rms_optimizer = RMSprop(lr=lr, epsilon=0.1, rho=0.99)

        self.epsilon = 0.1
        self.epsilon_increment = 0.01
        self.epsilon_max = 0.99

    def choose_action(self, obs):
        if self.is_discret:
            if np.random.uniform() < self.epsilon:
                action_value = self.model.predict(obs)
                action = np.argmax(action_value)
            else:
                action = np.random.randint(0, self.out_dim)
            return int(action)
        else:
            if np.random.uniform() < self.epsilon:
                action = self.model.predict(obs)
            else:
                action = self.model.predict(obs) + np.random.normal(size=self.out_dim)
            action = (np.max(action) - action) / (np.max(action) - np.min(action))
            return action[0]

    def optimizer(self):
        weighted_actions = K.sum(self.action_pl * self.model.output, axis=1)
        eligibility = K.log(weighted_actions + 1e-10) * K.stop_gradient(self.advantages_pl)
        entropy = K.sum(self.model.output * K.log(self.model.output + 1e-10), axis=1)
        loss = 0.001 * entropy - K.sum(eligibility)
        updates = self.rms_optimizer.get_updates(self.model.trainable_weights, [], loss)
        return K.function([self.model.input, self.action_pl, self.advantages_pl], [], updates=updates)

    def save(self):
        self.model.save_weights('../../results/actor_'+str(self.id)+'.h5')

    def load_weights(self):
        self.model.load_weights('../../results/actor_'+str(self.id)+'.h5')

    def build_model(self):
        inp = Input(shape=(self.inp_dim,))
        for i, layer in enumerate(self.layer):
            if i == 0:
                x = Dense(layer, activation='relu')(inp)
            else:
                x = Dense(layer, activation='relu')(x)
        if self.is_discret:
            out = Dense(self.out_dim, activation='softmax')(x)
            return Model(inp, out)
        else:
            out = Dense(self.out_dim, activation='tanh', kernel_initializer=RandomUniform())(x)
            return Model(inp, out)
