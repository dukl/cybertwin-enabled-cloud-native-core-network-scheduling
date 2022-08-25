import numpy as np
import copy

import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Concatenate
from keras.optimizers import Adam
import keras.backend as K
from collections import deque
import utils.global_parameters as GP
from agent.aiModels.FC import FC
from agent.aiModels.TensorStandardScaler import TensorStandardScaler
from tqdm import trange
import random
import os,sys
log_prefix = '[' + os.path.basename(__file__)
import utils.system_inner as SI
from utils.logger import log

class PNN():
    def __init__(self):
        self.sess = tf.Session()
        self.name = 'rodc'
        self.finalized = False
        self.layers, self.max_logvar, self.min_logvar = [], None, None
        self.decays, self.optvars, self.nonoptvars = [], [], []
        self.end_act, self.end_act_name = None, None
        self.scaler = None
        self.optimizer = None
        self.sy_train_in, self.sy_train_targ = None, None
        self.train_op, self.mse_loss = None, None
        self.sy_pred_in2d, self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = None, None, None
        self.sy_pred_mean2d, self.sy_pred_var2d = None, None
        self.sy_pred_in3d, self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = None, None, None
        self.num_nets = GP.n_ensemble
        self.model_loaded = False
        self.is_probabilistic = True
        self.is_tf_model = True
        self.state_dim, _ = SI.CHECK_ACT_OBS_DIM()
        self.state_dim -= 3
        self.act_dim = 1
        self.tf_sc     = tf.placeholder(shape=[1, self.state_dim], dtype=tf.float32)
        self.tf_A_avai = tf.placeholder(shape=[None, 1, self.act_dim], dtype=tf.float32)
        self.create_forward_model()
        self.memory = deque(maxlen=400000)
        self.train_in = np.array([]).reshape(
            0, self.state_dim+self.act_dim
        )
        self.train_target = np.array([]).reshape(
            0, self.state_dim
        )
        self.sess.run(tf.global_variables_initializer())


    def add(self, layer):
        layer.set_ensemble_size(self.num_nets)
        if len(self.layers) > 0:
            layer.set_input_dim(self.layers[-1].get_output_dim())
        self.layers.append(layer.copy())

    def finalize(self, optimizer, optimizer_args=None, *args, **kwargs):
        optimizer_args = {} if optimizer_args is None else optimizer_args
        self.optimizer = optimizer(**optimizer_args)
        self.layers[-1].set_output_dim(2*self.layers[-1].get_output_dim())
        self.end_act = self.layers[-1].get_activation()
        self.end_act_name = self.layers[-1].get_activation(as_func=False)
        self.layers[-1].unset_activation()

        with self.sess.as_default():
            with tf.variable_scope(self.name):
                self.scaler = TensorStandardScaler(self.layers[0].get_input_dim())
                self.max_logvar = tf.Variable(np.ones([1, self.layers[-1].get_output_dim() // 2])/2., dtype=tf.float32, name='max_log_var')
                self.min_logvar = tf.Variable(-np.ones([1, self.layers[-1].get_output_dim() // 2])*10., dtype=tf.float32, name='min_log_var')
                for i, layer in enumerate(self.layers):
                    with tf.variable_scope("Layer%i" % i):
                        layer.construct_vars()
                        self.decays.extend(layer.get_decays())
                        self.optvars.extend(layer.get_vars())
        self.optvars.extend([self.max_logvar, self.min_logvar])
        self.nonoptvars.extend(self.scaler.get_vars())

        with tf.variable_scope(self.name):
            self.optimizer = optimizer(**optimizer_args)
            self.sy_train_in = tf.placeholder(dtype=tf.float32, shape=[self.num_nets, None, self.layers[0].get_input_dim()], name='training_inputs')
            self.sy_train_targ = tf.placeholder(dtype=tf.float32, shape=[self.num_nets, None, self.layers[-1].get_output_dim()//2],name='training_targets')
            train_loss = tf.reduce_sum(self._compile_losses(self.sy_train_in, self.sy_train_targ, inc_var_loss=True))
            train_loss += tf.add_n(self.decays)
            train_loss += 0.01 * tf.reduce_sum(self.max_logvar) - 0.01 * tf.reduce_sum(self.min_logvar)
            self.mse_loss = self._compile_losses(self.sy_train_in, self.sy_train_targ, inc_var_loss=False)
            self.train_op = self.optimizer.minimize(train_loss, var_list=self.optvars)

        #self.sess.run(tf.variables_initializer(self.optvars+self.nonoptvars+self.optimizer.variables()))

        with tf.variable_scope(self.name):
            self.sy_pred_in2d = tf.placeholder(dtype=tf.float32, shape=[None, self.layers[0].get_input_dim()], name='2D_training_inputs')
            self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = self.create_prediction_tensors(self.sy_pred_in2d, factored=True)
            self.sy_pred_mean2d = tf.reduce_sum(self.sy_pred_mean2d_fac, axis=0)
            self.sy_pred_var2d = tf.reduce_sum(self.sy_pred_var2d_fac, axis=0) + tf.reduce_sum(tf.square(self.sy_pred_mean2d_fac-self.sy_pred_mean2d), axis=0)
            self.sy_pred_in3d = tf.placeholder(dtype=tf.float32, shape=[self.num_nets, None, self.layers[0].get_input_dim()], name='3D_training_inputs')
            self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = self.create_prediction_tensors(self.sy_pred_in3d, factored=True)
        self.finalized = True

    def train(self, batch_size):
        if len(self.memory) <= batch_size:
            return
        samples = random.sample(self.memory, batch_size)
        new_train_in, new_train_target = [], []
        for sample in samples:
            s_t, a_t, s_t1 = sample[0], sample[1], sample[2]
            new_train_in.append(np.concatenate([s_t, a_t], axis=-1))
            new_train_target.append(s_t1)
        self.train_in = np.concatenate([self.train_in]+new_train_in, axis=0)
        self.train_target = np.concatenate([self.train_target]+new_train_target, axis=0)
        self._train(self.train_in, self.train_target)

    def _train(self, inputs, targets, batch_size=32, epochs=100, hide_progress=False, holdout_ratio=0.0, max_logging=5000, misc=None):
        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:,None], idxs]
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        with self.sess.as_default():
            self.scaler.fit(inputs)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])
        if hide_progress:
            epoch_range = range(epochs)
        else:
            epoch_range = trange(epochs, unit="epoch(s)", desc="Network Training")
        for _ in epoch_range:
            for batch_num in range(int(np.ceil(idxs.shape[-1]/batch_size))):
                batch_idxs = idxs[:, batch_num*batch_size:(batch_num+1)*batch_size]
                self.sess.run(self.train_op, feed_dict={self.sy_train_in:inputs[batch_idxs], self.sy_train_targ:targets[batch_idxs]})
            idxs = shuffle_rows(idxs)
            if not hide_progress:
                if holdout_ratio < 1e-12:
                    epoch_range.set_postfix({"Training loss(es)": self.sess.run(self.mse_loss, feed_dict={self.sy_train_in:inputs[idxs[:,:max_logging]], self.sy_train_targ:targets[idxs[:,:max_logging]]})})
                else:
                    epoch_range.set_postfix({
                        "Training loss(es)": self.sess.run(
                            self.mse_loss,
                            feed_dict={
                                self.sy_train_in: inputs[idxs[:,:max_logging]],
                                self.sy_train_targ: targets[idxs[:,:max_logging]]
                            }
                        ),
                        "Holdout loss(es)": self.sess.run(
                            self.mse_loss,
                            feed_dict={
                                self.sy_train_in:holdout_inputs,
                                self.sy_train_targ:holdout_targets
                            }
                        )
                    })

    def predict(self, sc, A_avai, ts):
        log.logger.debug('before prediction, obs_env = \n%s' % (str(sc)))
        pred_obs = SI.NORM_STATE(copy.deepcopy(sc))
        pred_obs = [b for a in pred_obs for b in a]

        tf_pred_obs = self.tf_predict_state(len(A_avai))
        scV = np.array(pred_obs).reshape([1, self.state_dim])
        actV = []
        for action in A_avai:
            for act in action.value:
                for m, s, i, n in act:
                    act_value = s*GP.n_ms_server*(GP.ypi_max+1) + i * (GP.ypi_max+1) + n
                    actV.append(act_value)
        actV = np.array(actV)
        print(actV.shape)
        actV = actV.reshape(actV.shape[0], 1, 1)

        pred_state = self.sess.run(
            tf_pred_obs, feed_dict={self.tf_sc:scV, self.tf_A_avai:actV}
        )
        log.logger.debug('after prediction, pred_obs = \n%s' % (str(pred_state)))
        return pred_state[0].tolist()

    def create_prediction_tensors(self, inputs, factored=False, *args, **kwargs):
        factored_mean, factored_variance = self._compile_outputs(inputs)
        if inputs.shape.ndims == 2 and not factored:
            mean = tf.reduce_mean(factored_mean, axis=0)
            variance = tf.reduce_mean(tf.square(factored_mean - mean), axis=0) + tf.reduce_mean(factored_variance, axis=0)
            return mean, variance
        return factored_mean, factored_variance

    def _compile_outputs(self, inputs, ret_log_var=False, bm_train=False):
        dim_output = self.layers[-1].get_output_dim()
        cur_out = self.scaler.transform(inputs)
        i_layer_bm = 0
        for layer in self.layers:
            cur_out = layer.compute_output_tensor(cur_out, bm_train, name=str(i_layer_bm))
            i_layer_bm += 1
        mean = cur_out[:,:,:dim_output//2]
        if self.end_act is not None:
            mean = self.end_act(mean)
        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - cur_out[:,:,dim_output//2:])
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, tf.exp(logvar)

    def _compile_losses(self, inputs, targets, inc_var_loss=True):
        mean, log_var = self._compile_outputs(inputs, ret_log_var=True, bm_train=True)
        inv_var = tf.exp(-log_var)
        if inc_var_loss:
            mse_losses = tf.reduce_mean(tf.reduce_mean(tf.square(mean - targets) * inv_var, axis=-1), axis=-1)
            var_losses = tf.reduce_mean(tf.reduce_mean(log_var, axis=-1), axis=-1)
            total_losses = mse_losses + var_losses
        else:
            total_losses = tf.reduce_mean(tf.reduce_mean(tf.square(mean - targets), axis=-1), axis=-1)
        return total_losses

    def create_forward_model(self):
        self.add(FC(200, input_dim=self.state_dim+1, activation="swish", weight_decay=0.000025))
        self.add(FC(200, activation="swish", weight_decay=0.00005))
        self.add(FC(200, activation="swish", weight_decay=0.000075))
        self.add(FC(200, activation="swish", weight_decay=0.000075))
        self.add(FC(self.state_dim, weight_decay=0.0001))
        self.finalize(tf.train.AdamOptimizer, {"learning_rate": 0.001})

    def tf_predict_state(self, n):
        idx = tf.constant(0)
        def continue_prediction(idx, *args):
            return tf.less(idx, n)
        def iteration(idx, sc):
            cur_acs = self.tf_A_avai[idx]
            inputs = tf.concat([sc, cur_acs], axis=-1)
            mean, var = self.create_prediction_tensors(inputs)
            return idx + 1, mean + sc
        _, pred_state = tf.while_loop(
            cond=continue_prediction, body=iteration, loop_vars=[idx, self.tf_sc],
            shape_invariants=[
                idx.get_shape(), self.tf_sc.get_shape()
            ]
        )
        return pred_state