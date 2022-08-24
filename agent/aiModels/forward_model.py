from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from copy import deepcopy
import random
from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
import numpy as np
import utils.system_inner as SI
import copy
import utils.global_parameters as GP
from utils.logger import log

class FM:
    def __init__(self):
        self.obs_dim, self.act_dim = SI.CHECK_ACT_OBS_DIM()
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(24, input_dim=self.obs_dim - 3 + 1, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.obs_dim - 3, activation='softmax'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        return model

    def predict(self, obs_env, A_avai, ts):
        log.logger.debug('before prediction, obs_env = \n%s' % (str(obs_env)))
        pred_obs = SI.NORM_STATE(copy.deepcopy(obs_env))
        pred_obs = [b for a in pred_obs for b in a]

        for action in A_avai:
            for act in action.value:
                for m, s, i, n in act:
                    act_value = s*GP.n_ms_server*(GP.ypi_max+1) + i * (GP.ypi_max+1) + n
                    pred_obs = np.array(pred_obs + [act_value])
                    pred_obs  = self.model.predict(np.reshape(pred_obs, [1, pred_obs.shape[0]]))
                    pred_obs = pred_obs[0].tolist()

        pred_obs = np.array(pred_obs).reshape(int(len(pred_obs)/2), 2)

        for m in range(len(GP.c_r_ms)):
            for n in range(GP.n_servers):
                for i in range(GP.n_ms_server):
                    idx = m * GP.n_servers * GP.n_ms_server + n * GP.n_ms_server + i
                    # log.logger.debug('lamda=%f, n_threads=%f' % (obs_env[idx][0], obs_env[idx][1]))
                    pred_obs[idx][0] = int(pred_obs[idx][0]*GP.lamda_ms[m])
                    pred_obs[idx][1] = int(pred_obs[idx][1]*GP.ypi_max)
        log.logger.debug('after prediction, pred_obs = \n%s' % (str(pred_obs)))
        return pred_obs