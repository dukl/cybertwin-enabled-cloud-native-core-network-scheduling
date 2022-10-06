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
from agent.aiModels.state_predictor import SP

class TSAND:
    def __init__(self):
        self.sess = tf.Session()
        self.obs_dim = ACS.n_node * 8
        self.actors = []
        self.critic = None
        self._build_actor_critic_networks()
        self.a_update_steps = 5
        self.c_update_steps = 5

        self.predictor = SP(self.sess, self.obs_dim, 3+ACS.n_max_inst+2+ACS.n_max_inst*ACS.n_max_inst, [256, 256, 128])

        self.sess.run(tf.global_variables_initializer())

        self.sc = np.random.uniform(size=self.obs_dim)
        self.sc_id = -1
        self.act_seq = []

    def _build_actor_critic_networks(self):
        for i, params in enumerate(ACS.actors):
            if params[1] is True: # discrete action
                actor = APPO_D(i, self.sess, params[0], self.obs_dim, params[2], layers=params[3])
            else:
                actor = APPO_C(i, self.sess, params[0], self.obs_dim, params[2], layers=params[3])
            self.actors.append(actor)
        self.critic = CPPO(self.sess, self.obs_dim, ACS.critic[0], layers=ACS.critic[1])

    def update(self, s, a, r):
        #log.logger.debug('Training ...')
        #print('%s' % (str(r.tolist())))
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
        ##log.logger.debug('action-0 %s, shape=%s' % (str(ba[0]), str(ba[0].shape)))

        for i, actor in enumerate(self.actors):
            print('dukl-actor-%d' % (i))
            self.sess.run(actor.update_oldpi_op)
            adv = self.sess.run(self.critic.advantage, {self.critic.tfs: s, self.critic.tfdc_r: r})
            #adv = (adv - np.mean(adv)) / np.std(adv)
            print('avd: %s' % (str(adv)))
            ##print(self.sess.run(actor.a_indices, {actor.tfa: ba[i]}))
            for _ in range(self.a_update_steps):
                #if i == 6:
                print('ratio = \n',(self.sess.run(actor.ratio, {actor.tfs: s, actor.tfa: ba[i], actor.tfadv: adv})).tolist())
                    #print('oldpi: ',(self.sess.run(actor.oldpi.prob, {actor.tfs: s, actor.tfa: ba[i], actor.tfadv: adv})).tolist())
                print('loss = \n',self.sess.run(actor.aloss, {actor.tfs: s, actor.tfa: ba[i], actor.tfadv: adv}))
                self.sess.run(actor.atrain_op, {actor.tfs: s, actor.tfa: ba[i], actor.tfadv: adv})
            #[self.sess.run(actor.atrain_op, {actor.tfs: s, actor.tfa: ba[i], actor.tfadv: adv}) for _ in range(self.a_update_steps)]
        [self.sess.run(self.critic.ctrain_op, {self.critic.tfs: s, self.critic.tfdc_r: r}) for _ in range(self.c_update_steps)]

    def choose_actions(self, obs):
        action = ACTIONS()
        act_value = []
        for i, actor in enumerate(self.actors):
            act_value.append(actor.choose_action(obs))
            ##log.logger.debug('action-%d = %s' % (i, str(act_value[-1])))
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

    def random_chosn_an_action(self):
        action = ACTIONS()
        action.scale_in_out_idx = np.random.randint(0,3) - 1
        action.scale_nf_idx = np.random.randint(0, len(ACS.t_NFs) - 1) + 1
        action.scale_in_out_node_idx = np.random.randint(0, ACS.n_node)
        action.scale_up_dn = np.random.random(ACS.n_max_inst)
        action.scheduling_src_nf_idx = np.random.randint(0, len(ACS.t_NFs) - 1) + 1
        action.scheduling_dst_nf_idx = np.random.randint(0, len(ACS.t_NFs) - 1) + 1
        action.scheduling = np.random.random(ACS.n_max_inst * ACS.n_max_inst)
        return action

    def choose_action_with_delayed_obs(self, obs_on_road, ts):
        avai_obs, valid_obs = [], []
        for i, obs in enumerate(obs_on_road):
            if obs[0] + obs[1] < ts: # obs travels at the agent
                avai_obs.append(obs)
        for obs in avai_obs:
            obs_on_road.remove(obs)
        #log.logger.debug('receiving %d obs (%d obs on road)' % (len(avai_obs), len(obs_on_road)))
        if self.sc_id == -1 and len(self.act_seq) == 0:
            self.act_seq.append([-1, self.random_chosn_an_action()])
        else:
            if len(avai_obs) > 0:
                for i, obs in enumerate(avai_obs):
                    if obs[0] > self.sc_id:
                        valid_obs.append(obs)
                if len(valid_obs) > 0:
                    valid_obs.sort(key=lambda x:(x[0]), reverse=True)
                    self.sc = valid_obs[0][2]
                    self.sc_id = valid_obs[0][0]
            for act in self.act_seq:
                if act[0] < self.sc_id or act[0] >= ts:
                    self.act_seq.remove(act)
        #log.logger.debug('predicting s[%d] using %d pending actions and sc' % (ts, len(self.act_seq)))
        #obs_pred =
