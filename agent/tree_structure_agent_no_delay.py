from tqdm import tqdm
#from keras.models import Model
#from keras import regularizers
#from keras.utils import to_categorical
#from keras.layers import Input, Dense, Flatten
#from keras.initializers import RandomUniform
import numpy as np
from utils.logger import log
import utils.auto_scaling_settings as ACS
from results.metrics import metrics
from utils.actions_definition import ACTIONS
#import keras.backend as K
#from keras.optimizers import RMSprop
#import tensorflow.compat.v1 as tf
#tf.compat.v1.disable_eager_execution()
import tensorflow as tf
from agent.aiModels.PPO_actor_discrete import APPO_D
from agent.aiModels.PPO_actor_contiuous import APPO_C
from agent.aiModels.PPO_critic import CPPO

class TSAND:
    def __init__(self):
        self.sess = tf.Session()
        self.obs_dim = ACS.n_node * 8
        self.actors = []
        self.critic = None
        self._build_actor_critic_networks()
        self.a_update_steps = 10
        self.c_update_steps = 10
        self.sess.run(tf.global_variables_initializer())

    def _build_actor_critic_networks(self):
        for i, params in enumerate(ACS.actors):
            if params[1] is True: # discrete action
                actor = APPO_D(i, self.sess, params[0], self.obs_dim, params[2], layers=params[3])
            else:
                actor = APPO_C(i, self.sess, params[0], self.obs_dim, params[2], layers=params[3])
            self.actors.append(actor)
        self.critic = CPPO(self.sess, self.obs_dim, ACS.critic[0], layers=ACS.critic[1])

    def update(self, s, a, r):
        log.logger.debug('Training ...')
        action = [[] for _ in range(len(ACS.actors))]
        ba = [[] for _ in range(len(ACS.actors))]
        for i in range(len(a)):
            action[0].append(a[i].scale_in_out_idx + 1)
            action[1].append(a[i].scale_nf_idx - 1)
            action[2].append(a[i].scale_in_out_node_idx)
            action[3].append(a[i].scale_up_dn)
            action[4].append(a[i].scheduling_src_nf_idx - 1)
            action[5].append(a[i].scheduling_dst_nf_idx - 1)
            action[6].append(a[i].scheduling)
        for i in range(len(ACS.actors)):
            if ACS.actors[i][1] is True:
                ba[i] = np.vstack(action[i]).ravel()
            else:
                ba[i] = np.vstack(action[i])
        log.logger.debug('action-0 %s, shape=%s' % (str(ba[0]), str(ba[0].shape)))

        for i, actor in enumerate(self.actors):
            self.sess.run(actor.update_oldpi_op)
            adv = self.sess.run(self.critic.advantage, {self.critic.tfs: s, self.critic.tfdc_r: r})
            #print(self.sess.run(actor.a_indices, {actor.tfa: ba[i]}))
            [self.sess.run(actor.atrain_op, {actor.tfs: s, actor.tfa: ba[i], actor.tfadv: adv}) for _ in range(self.a_update_steps)]
        [self.sess.run(self.critic.ctrain_op, {self.critic.tfs: s, self.critic.tfdc_r: r}) for _ in range(self.c_update_steps)]

    def choose_actions(self, obs):
        action = ACTIONS()
        act_value = []
        for i, actor in enumerate(self.actors):
            act_value.append(actor.choose_action(obs))
            #log.logger.debug('action-%d = %s' % (i, str(act_value[-1])))
        action.scale_in_out_idx = act_value[0] - 1
        metrics.value[-1].action_ops = int(act_value[0]) - 1
        action.scale_nf_idx = act_value[1] + 1
        metrics.value[-1].action_nf_select = int(act_value[1]) + 1
        action.scale_in_out_node_idx = act_value[2]
        metrics.value[-1].actor_node_select = int(act_value[2])
        action.scale_up_dn = act_value[3].tolist()
        action.scheduling_src_nf_idx = act_value[4] + 1
        metrics.value[-1].action_src_nf_select = int(act_value[4]) + 1
        action.scheduling_dst_nf_idx = act_value[5] + 1
        metrics.value[-1].action_dst_nf_select = int(act_value[5]) + 1
        action.scheduling = act_value[6].tolist()
        return action