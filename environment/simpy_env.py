import math
import random, simpy, numpy

import numpy as np

import utils.auto_scaling_settings as ACS
from utils.logger import log, log_d
from environment.flowsimulator import FlowSimulator
from utils.actions_definition import ACTIONS
from results.metrics import metrics, VALUE
from agent.tree_structure_agent_no_delay import TSAND
from agent.hybrid_agent_d_c import HADC
#from agent.maddpg import MADDPG
from agent.maddpg_agent import MADDPG_NEW
from agent.aiModels.ddpg import DDPG
from agent.signal_agent_ppo import PPO
#from agent.torch_ddpg import DDPG
from agent.aiModels.utils import create_directory, plot_learning_curve, scale_action
from agent.aiModels.dqn_agents import DDQNAgent
from agent.dqn_agents import Agent

EP_MAX, EP_LEN, BATCH, GAMMA = 500, 100, 100, 0.9
TIME_PER_STEP = 100
NF_CPU = 300
NODE_CPU = 200000
N_INST = 4
NCOM = 1
N_CAPA = 5000
N_AMF_INST = 1

class ConditionalGet(simpy.resources.base.Get):
    def __init__(self, resource, condition=lambda: True):
        self.condition = condition
        super().__init__(resource)
class aStore(simpy.resources.store.Store):
    get = simpy.core.BoundClass(ConditionalGet)
    def _do_get(self, event):
        if event.condition():
            super()._do_get(event)

class NODE:
    def __init__(self, id):
        self.id = id
        self.remaining_resource = NODE_CPU # [CPU, ...]
        self.maximum_resource = NODE_CPU
        self.nfs = [[] for _ in range(len(ACS.t_NFs))] # how many instances for each service
        #self.scheduling_table = np.zeros(len(ACS.msg_msc), len(ACS.t_NFs) - 1)

    def get_remain_resources(self, nf_id, inst_id):
        sum = 0
        for nf in self.nfs:
            for inst in nf:
                if inst.is_alive == False:
                    continue
                if inst.id == nf_id and inst.inst_id == inst_id:
                    continue
                sum += inst.allocated_resource
        return self.maximum_resource - sum

    def get_resources_used(self):
        sum = 0
        for nf in self.nfs:
            for inst in nf:
                if inst.is_alive == False:
                    continue
                sum += inst.allocated_resource
        return sum

class NF:
    def __init__(self, sim_env, env, id, inst_id, loc_id):
        self.message_queue      = aStore(sim_env, capacity=simpy.core.Infinity)#simpy.Store(env=sim_env, capacity=simpy.core.Infinity) #
        self.id = id
        self.loc_id = loc_id
        self.inst_id = inst_id
        self.env = env
        self.sim_env = sim_env
        #self.inst_index = [0 for _ in range(len(ACS.t_NFs))]
        self.n_flow_process = [0 for _ in range(len(ACS.t_NFs))]
        self.weight_load = [[] for _ in range(len(ACS.t_NFs))]
        self.is_alive_event = self.sim_env.event()
        self.is_alive = False
        self.scheduling_table = [[] for _ in range(len(ACS.t_NFs))]
        self.global_id = self.inst_id + self.loc_id
        self.allocated_resource = NF_CPU #+ 100 + self.global_id*200#300#200 + self.global_id*100  # random.randint(400,800)
        self.max_load = N_CAPA#100 + self.global_id*100 #random.randint(200, 250)

    def determize_next_nf_inst_new(self, nnf_id, flow):
        #log.logger.debug('flow-%d-%d-%d len(chain)=%d, index=%d' % (flow.ue_id, flow.pro_id, flow.msg_id, len(flow.node_chain), flow.index - 1))
        nnf_loc_id = flow.node_chain[flow.index - 1]
        avai_inst = self.env.check_node_nf_insts(nnf_loc_id, nnf_id)
        min_ratio = 1000
        ret_inst = None
        #log.logger.debug('avai_inst = %d' % (len(avai_inst)))
        for i, inst in enumerate(avai_inst):
            #log.logger.debug('ratio = %f' % (len(inst.message_queue.items)/inst.allocated_resource))
            if len(inst.message_queue.items) / inst.allocated_resource < min_ratio:
                min_ratio = len(inst.message_queue.items) / inst.allocated_resource
                ret_inst = inst
        #log.logger.debug('choose nf-%d-%d-%d' % (ret_inst.loc_id, ret_inst.id, ret_inst.inst_id))
        return ret_inst

    def determize_next_nf_inst(self, next_nf_id):
        next_nf_inst_idx, n_sum = 0, 0
        ##log.logger.debug('[NF-%d-%d-%d] avai_insts for from nf-%d to nf-%d is %d; table=%s' % (self.loc_id, self.id, self.inst_id, self.id, next_nf_id, len(self.weight_load[next_nf_id]), str(self.weight_load[next_nf_id])))
        for i in range(len(self.weight_load[next_nf_id])):
            ###log.logger.debug('[NF-%d-%d-%d] weight for %d-th inst = %d, total=%d, index=%d, sum=%d' % (self.loc_id, self.id, self.inst_id, i, self.weight_load[next_nf_id][i], sum(self.weight_load[next_nf_id]), self.n_flow_process[next_nf_id], n_sum))
            n_sum += self.weight_load[next_nf_id][i]
            if self.n_flow_process[next_nf_id] < n_sum:
                next_nf_inst_idx = i
                ##log.logger.debug('[NF-%d-%d-%d] choose %d-th instance' % (self.loc_id, self.id, self.inst_id, next_nf_inst_idx))
                break
        if len(self.weight_load[next_nf_id]) == 0:
            #log.logger.debug('[NF-%d-%d-%d] avai_insts for from nf-%d to nf-%d is %d; table=%s' % (self.loc_id, self.id, self.inst_id, self.id, next_nf_id, len(self.weight_load[next_nf_id]), str(self.weight_load[next_nf_id])))
            return None
        self.n_flow_process[next_nf_id] = (self.n_flow_process[next_nf_id] + 1) % sum(self.weight_load[next_nf_id])
        return self.scheduling_table[next_nf_id][next_nf_inst_idx]

    def handle_flow(self, simpy_env):
        #if self.id == 0: # RISE handling
        while True:
            yield self.is_alive_event
            #log.logger.debug('[NF-%d-%d-%d][res = %f] handle_flow in nf-%d-%d-%d: time = %f' % (self.loc_id, self.id, self.inst_id, self.allocated_resource, self.loc_id, self.id, self.inst_id, self.sim_env.now))
            #if self.loc_id == 0 and self.id == 3 and self.inst_id == 0:
            #    log.logger.debug('before nf-%d-%d-%d has load %d time=%f' % (self.loc_id, self.id, self.inst_id, len(self.message_queue.items), self.sim_env.now))
            flow = yield self.message_queue.get()
            #if self.loc_id == 0 and self.id == 3 and self.inst_id == 0:
                #log.logger.debug('after nf-%d-%d-%d has load %d time=%f' % (self.loc_id, self.id, self.inst_id, len(self.message_queue.items), self.sim_env.now))
            #log.logger.debug('nf-%d-%d-%d is handling flow-%d-%d-%d-%d: time = %f' % (self.loc_id, self.id, self.inst_id, flow.ue_id, flow.pro_id, flow.msg_id, flow.index, self.sim_env.now))
            if ACS.flow_ttl[flow.msg_id] < (self.sim_env.now - flow.creation_time):
                #log.logger.debug('flow-%d-%d-%d-%d timeout (%f)' % (flow.ue_id, flow.pro_id, flow.msg_id, flow.index, (self.sim_env.now - flow.creation_time)))
                metrics.value[-1].n_fail_reqs += 1
                metrics.value[-1].n_timeout += 1
                continue
            flow.index += 1
            #log.logger.debug('flow-%d-%d-%d len(chain)=%d, index=%d, len(msc)=%d' % (flow.ue_id, flow.pro_id, flow.msg_id, len(flow.node_chain), flow.index, len(ACS.msg_msc[flow.msg_id])))
            if flow.index >= len(ACS.msg_msc[flow.msg_id]):
                yield self.sim_env.timeout(NCOM*ACS.computing_demand[flow.msg_id][flow.index-1]/self.allocated_resource)  # processing time
                metrics.value[-1].n_succ_reqs += 1
                metrics.value[-1].average_delay += (self.sim_env.now - flow.creation_time)
                metrics.value[-1].qos_delay += (1 - (self.sim_env.now - flow.creation_time)/ACS.flow_ttl[flow.msg_id])
                #log.logger.debug('flow-%d-%d-%d has been processed successfully execution time = [%f-%f]' % (flow.ue_id, flow.pro_id, flow.msg_id, flow.creation_time, self.sim_env.now))
                #log.logger.debug('average delay for flow-%d-%d-%d is %f' % (flow.ue_id, flow.pro_id, flow.msg_id, self.sim_env.now - flow.creation_time))
                continue
            src_nf_inst_loc = self.loc_id
            next_nf = ACS.msg_msc[flow.msg_id][flow.index]

            next_nf_inst = self.determize_next_nf_inst_new(next_nf, flow)
            if next_nf_inst is None:
                metrics.value[-1].n_fail_reqs += 1 # no avai instances in this node
                log.logger.debug('no available nf for flow-%d-%d-%d, chain = %s, nfs=%s' % (flow.ue_id, flow.pro_id, flow.msg_id, str(flow.node_chain), str(ACS.msg_msc[flow.msg_id])))
                continue

            #self.n_flow_process[next_nf] = (self.n_flow_process[next_nf]+1) % sum(self.weight_load[next_nf])
            #next_nf_inst = (self.inst_index[next_nf] + 1) % len(self.env.nfs[next_nf]) # round robin (RR)
            #dst_nf_inst_loc = self.env.nfs[next_nf][next_nf_inst].loc_id # which instance
            dst_nf_inst_loc = next_nf_inst.loc_id
            flow.nf_inst_id = next_nf_inst.inst_id
            #self.inst_index[next_nf] = next_nf_inst
            yield self.sim_env.timeout(NCOM*ACS.computing_demand[flow.msg_id][flow.index - 1] / self.allocated_resource)  # processing time
            #if self.id == 0:
            #    yield self.sim_env.timeout(ACS.computing_demand[flow.msg_id][flow.index-1]/self.allocated_resource)  # processing time
            #else:
            #    yield self.sim_env.timeout(ACS.computing_demand[flow.msg_id][flow.index-1]/self.allocated_resource)  # processing time
            #log.logger.debug('send flow-%d-%d-%d from nf-%d-%d-%d to nf-%d-%d-%d, msg_on_road-%d-%d: leaving at time %f' % (flow.ue_id, flow.pro_id, flow.msg_id, src_nf_inst_loc, self.id, self.inst_id, dst_nf_inst_loc, next_nf, next_nf_inst.inst_id, src_nf_inst_loc, dst_nf_inst_loc, self.sim_env.now))
            flow.in_msg_on_road_time = self.sim_env.now
            if src_nf_inst_loc == dst_nf_inst_loc:
                if len(self.env.nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id * ACS.n_node + dst_nf_inst_loc].message_queue.items) + 1 > self.env.nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id * ACS.n_node + dst_nf_inst_loc].max_load:  # ACS.n_max_load_in_nf[ACS.msg_msc[flow.msg_id][flow.index]]:
                    metrics.value[-1].n_fail_reqs += 1
                    metrics.value[-1].n_overload += 1
                    #log.logger.debug('nf-%d-%d-%d overload' % (
                    #self.env.nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id * ACS.n_node + dst_nf_inst_loc].loc_id,
                    #self.env.nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id * ACS.n_node + dst_nf_inst_loc].id,
                    #self.env.nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id * ACS.n_node + dst_nf_inst_loc].inst_id))
                else:
                    yield self.env.nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id * ACS.n_node + dst_nf_inst_loc].message_queue.put(flow)
                #if self.loc_id == 0 and self.id == 3 and self.inst_id == 0:
                #    log.logger.debug('flow-%d-%d-%d arrival nf-%d-%d-%d at time %f' % (flow.ue_id, flow.pro_id, flow.msg_id, self.env.nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id*ACS.n_node+dst_nf_inst_loc].loc_id, ACS.msg_msc[flow.msg_id][flow.index], self.env.nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id*ACS.n_node+dst_nf_inst_loc].inst_id, self.sim_env.now))
            else:
                #yield self.sim_env.timeout(0.001)
                yield self.env.msg_on_road[src_nf_inst_loc][dst_nf_inst_loc].put(flow)
        #else: # other NFs handling
        #    pass


class ENV_SIMPY:
    def __init__(self, sim_env):
        self.sim_env = sim_env
        self.nodes = []
        self.nodes_delay_map = [[0 for _ in range(ACS.n_node)] for _ in range(ACS.n_node)]
        self.msg_on_road = [[None for _ in range(ACS.n_node)] for _ in range(ACS.n_node)]
        self.nfs = [[] for _ in range(len(ACS.t_NFs))]
        self._initial_topo()

    def _initial_topo(self):
        for i in range(ACS.n_node):
            self.nodes.append(NODE(i))
        for i in range(len(self.nodes)):
            for j in range(len(self.nodes)):
                self.msg_on_road[i][j] = aStore(self.sim_env, capacity=simpy.core.Infinity)#simpy.Store(env=self.sim_env, capacity=simpy.core.Infinity) #
                if i == j:
                    continue
                else:
                    self.nodes_delay_map[i][j] = 1#random.random()*ACS.max_delay_between_nodes
        ##log.logger.debug('delays = \n%s' % (str(numpy.array(self.nodes_delay_map))))
        for i in range(len(ACS.t_NFs)):
            for j in range(ACS.n_max_inst):
                for loc_id in range(ACS.n_node):
                    nf = NF(self.sim_env, self, ACS.t_NFs[i], j, loc_id)
                    #nf.allocated_resource = 100#random.random()*400 + 100
                    self.nfs[i].append(nf)
                    self.nodes[loc_id].nfs[i].append(nf)
                    #if j == 0:
                        #print('???')
                        #self.nfs[i][loc_id].is_alive = True
                        #self.nfs[i][loc_id].is_alive_event.succeed()
                        #log.logger.debug('nf-%d-%d-%d evoked' % (self.nfs[i][loc_id].loc_id, self.nfs[i][loc_id].id, self.nfs[i][loc_id].inst_id))
                    if i != 1 and j < N_INST:
                        nf.is_alive = True
                        nf.is_alive_event.succeed()
                        log.logger.debug('nf-%d-%d-%d evoked (CPU = %d)' % (nf.loc_id, nf.id, nf.inst_id, nf.allocated_resource))
                    if i == 1 and j < N_AMF_INST:
                        nf.is_alive = True
                        nf.is_alive_event.succeed()
                        log.logger.debug(
                            'nf-%d-%d-%d evoked (CPU = %d)' % (nf.loc_id, nf.id, nf.inst_id, nf.allocated_resource))
                #loc_id = random.randint(0,ACS.n_node-1)
                #self.nfs[i][j*ACS.n_node + loc_id].is_alive = True
                #self.nfs[i][j*ACS.n_node + loc_id].is_alive_event.succeed()
                #log.logger.debug('nf-%d-%d-%d evoked' % (self.nfs[i][j*ACS.n_node + loc_id].loc_id, self.nfs[i][j*ACS.n_node + loc_id].id, self.nfs[i][j*ACS.n_node + loc_id].inst_id))
            #inst_id = 0#random.randint(0, ACS.n_max_inst-1)
            #loc_id  = 0#random.randint(0, ACS.n_node - 1)
            #self.nfs[i][inst_id*ACS.n_node + loc_id].is_alive_event.succeed()
            #self.nfs[i][inst_id*ACS.n_node + loc_id].is_alive = True
            #log.logger.debug('nf-%d-%d-%d evoked' % (self.nfs[i][inst_id*ACS.n_node + loc_id].loc_id, self.nfs[i][inst_id*ACS.n_node + loc_id].id, self.nfs[i][inst_id*ACS.n_node + loc_id].inst_id))
            ##log.logger.debug('nf-%d-%d-%d evoked' % (self.nfs[i][0].loc_id, self.nfs[i][0].id, self.nfs[i][0].inst_id))
        #self.update_scheculing_table()

    def rest(self):
        self.nodes = []
        self.nfs = [[] for _ in range(len(ACS.t_NFs))]

    def update_scheculing_table(self):
        for t in ACS.t_NFs:
            for nf in self.nfs[t]:
                if nf.is_alive == False:
                    continue
                for nnf in ACS.app[t]:
                    nf.scheduling_table[nnf].clear()
                    nf.weight_load[nnf].clear()
                    nf.scheduling_table[nnf] = self.check_available_instance_by_nf_id(nnf)
                    for _ in range(len(nf.scheduling_table[nnf])):
                        nf.weight_load[nnf].append(1)

    def check_available_instances(self, nf_id, loc_id):
        unavai_insts = []
        for inst in self.nodes[loc_id].nfs[nf_id]:
            if inst.is_alive == False:
                unavai_insts.append(inst)
        return unavai_insts

    def check_available_instance_by_nf_id(self, nf_id):
        avai_insts = []
        for inst in self.nfs[nf_id]:
            if inst.is_alive == True:
                avai_insts.append(inst)
        return avai_insts

    def add_new_inst(self, added_inst):
        remain_resource_in_node = self.nodes[added_inst.loc_id].get_resources_used()
        if  remain_resource_in_node + added_inst.allocated_resource >= self.nodes[added_inst.loc_id].maximum_resource:
            ##log.logger.debug('Punnish? .. adding inst failed, because exceed maximum resource of node-%d (%f)' % (added_inst.loc_id, remain_resource_in_node))
            return
        added_inst.is_alive = True
        added_inst.is_alive_event.succeed()
        for t in ACS.t_NFs:
            if t == added_inst.id:
                continue
            for inst in self.nfs[t]:
                if inst.is_alive == False:
                    continue
                ##log.logger.debug('nf-%d-%d-%d is adding nf-%d-%d-%d' % (inst.loc_id, inst.id, inst.inst_id, added_inst.loc_id, added_inst.id, added_inst.inst_id))
                if inst.id == 0: # RISE
                    inst.weight_load[added_inst.id].append(1)
                    inst.scheduling_table[added_inst.id].append(added_inst)
                else:
                    if added_inst.id in ACS.app[inst.id]:
                        inst.weight_load[added_inst.id].append(1)
                        inst.scheduling_table[added_inst.id].append(added_inst)
                    if inst.id in ACS.app[added_inst.id]:
                        added_inst.weight_load[t].append(1)
                        added_inst.scheduling_table[t].append(inst)
        #log.logger.debug('nf-%d-%d-%d evoked' % (added_inst.loc_id, added_inst.id, added_inst.inst_id))
    def delete_one_inst(self, delated_inst):
        delated_inst.is_alive = False
        delated_inst.is_alive_event = self.sim_env.event()
        delated_inst.scheduling_table = [[] for _ in range(len(ACS.t_NFs))]
        delated_inst.weight_load = [[] for _ in range(len(ACS.t_NFs))]
        delated_inst.allocated_resource = NF_CPU
        for t in ACS.t_NFs:
            if t == delated_inst.id:
                continue
            for inst in self.nfs[t]:
                if inst.is_alive == False:
                    continue
                ##log.logger.debug('nf-%d-%d-%d is deleting nf-%d-%d-%d' % (inst.loc_id, inst.id, inst.inst_id, delated_inst.loc_id, delated_inst.id, delated_inst.inst_id))
                if delated_inst.id in ACS.app[inst.id]:
                    idx = inst.scheduling_table[delated_inst.id].index(delated_inst)
                    del inst.weight_load[delated_inst.id][idx]
                    del inst.scheduling_table[delated_inst.id][idx]
                    #log.logger.debug('release connection between nf-%d-%d-%d and nf-%d-%d-%d' % (inst.loc_id, inst.id, inst.inst_id, delated_inst.loc_id, delated_inst.id, delated_inst.inst_id))
        #log.logger.debug('nf-%d-%d-%d closed' % (delated_inst.loc_id, delated_inst.id, delated_inst.inst_id))
        avai_insts = self.check_available_instance_by_nf_id(delated_inst.id)
        self.sim_env.process(self.moving_flows(delated_inst, avai_insts))

    def moving_flows(self, deleted_nf, same_nf):
        ##log.logger.debug('moving flows at time %f' % (self.sim_env.now))
        n_processed_msg = len(deleted_nf.message_queue.items)
        index = 0
        for _ in range(n_processed_msg):
            if len(deleted_nf.message_queue.items) > 0:
                flow = yield deleted_nf.message_queue.get()
                n_times = 0
                is_moved = True
                while True:
                    n_times += 1
                    if len(same_nf[index].message_queue.items) + 1 <= same_nf[index].max_load:
                        yield same_nf[index].message_queue.put(flow)
                        break
                    else:
                        index = (index + 1) % len(same_nf)
                    if n_times >= len(same_nf):
                        is_moved = False
                        break
                if is_moved == False:
                    metrics.value[-1].n_fail_reqs += 1
                    metrics.value[-1].n_overload += 1
                index = (index + 1) % len(same_nf)

    def check_node_nf_insts(self, loc_id, nf_id):
        avai_insts = []
        for inst in self.nodes[loc_id].nfs[nf_id]:
            if inst.is_alive == True:
                avai_insts.append(inst)
        return avai_insts

    def execute_new_action(self, action, simulator):
        for i in range(ACS.n_node):
            for j in range(len(ACS.t_NFs)-1):
                #avai_insts = self.check_available_instance_by_nf_id(j+1)
                unavai_insts = self.check_available_instances(j+1, i)
                n_inst = ACS.n_max_inst - len(unavai_insts)
                if action.h_s[i,j] == -1:
                    if n_inst == 1:
                        metrics.value[-1].punish_reward += 1
                        #log.logger.debug('at least one instance')
                        pass
                    elif n_inst - 1 < 0:
                        log.logger.debug('no active instance')
                        pass
                    else:
                        log.logger.debug('close this instance')
                        nf_instances = list(set(self.nodes[i].nfs[j+1]) - set(unavai_insts))
                        nf_instances.sort(key=lambda NF: len(NF.message_queue.items), reverse=False)
                        ##log.logger.debug('deleting an instance nf-%d-%d-%d' % (nf_instances[0].loc_id, nf_instances[0].id, nf_instances[0].inst_id))
                        nf_instances[0].is_alive = False
                        nf_instances[0].is_alive_event = self.sim_env.event()
                        nf_instances[0].allocated_resource = NF_CPU
                        #log.logger.debug('nf-%d-%d-%d closed' % (nf_instances[0].loc_id, nf_instances[0].id, nf_instances[0].inst_id))
                        avai_insts = self.check_available_instance_by_nf_id(nf_instances[0].id)
                        self.sim_env.process(self.moving_flows(nf_instances[0], avai_insts))
                if action.h_s[i,j] == 0:
                    #log.logger.debug('maintain')
                    pass
                if action.h_s[i,j] == 1:
                    #log.logger.debug('add one instance')
                    if n_inst + 1 > ACS.n_max_inst:
                        metrics.value[-1].punish_reward += 1
                        #log.logger.debug('exceed maximum instance at this node')
                        pass
                    else:
                        node_res_remain = self.nodes[i].get_resources_used()
                        if node_res_remain + NF_CPU <= self.nodes[i].maximum_resource:
                            idx = random.randint(0, len(unavai_insts) - 1)
                            unavai_insts[idx].is_alive = True
                            unavai_insts[idx].is_alive_event.succeed()
                            log.logger.debug('nf-%d-%d-%d evoked' % (unavai_insts[idx].loc_id, unavai_insts[idx].id, unavai_insts[idx].inst_id))
                        else:
                            metrics.value[-1].punish_reward += 1

                #for k, inst in enumerate(self.nodes[i].nfs[j+1]):
                #    if inst.is_alive == True:
                #        cpu = action.v_s[i,j*ACS.n_max_inst+k] * self.nodes[i].maximum_resource * 0.8
                #        log.logger.debug('nf-%d-%d-%d is allocated %f cpu' % (inst.loc_id, inst.id, inst.inst_id, cpu))
                #        if cpu < NF_CPU:
                #            continue
                #        inst.allocated_resource = cpu

                nf_instances = list(set(self.nodes[i].nfs[j + 1]) - set(unavai_insts))
                for n, inst in enumerate(nf_instances):
                    node_res_remain = self.nodes[inst.loc_id].get_remain_resources(inst.id, inst.inst_id)
                    cpu = inst.allocated_resource + 100 * (action.v_s[i * (len(ACS.t_NFs) - 1) + j][n])
                    if cpu > node_res_remain:
                        #log.logger.debug('exceed maximum resources for node-%d' % (inst.loc_id))
                        #metrics.value[-1].punish_reward += 1
                        continue
                    if cpu < NF_CPU:
                        #metrics.value[-1].punish_reward += 1
                        #log.logger.debug('lower than 300')
                        continue
                    inst.allocated_resource = cpu
                    #log.logger.debug('nf-%d-%d-%d is allocated %f cpu' % (inst.loc_id, inst.id, inst.inst_id, inst.allocated_resource))

        for i in range(len(ACS.t_NFs)-1):
            sum_cpu = 0
            for j in range(ACS.n_node):
                 for inst in self.nodes[j].nfs[i+1]:
                     if inst.is_alive == True:
                        sum_cpu += inst.allocated_resource
            for j in range(ACS.n_node):
                sum = 0
                for inst in self.nodes[j].nfs[i + 1]:
                    if inst.is_alive == True:
                        sum += inst.allocated_resource
                simulator.scheduling_table[i, j] = sum / sum_cpu * 10
        simulator.scheduling_table = simulator.scheduling_table.astype('int32')
        #print(simulator.scheduling_table)

        #simulator.scheduling_table = action.sch
        #tmp = np.sum(action.sch, 1).reshape(-1, 1)
        #simulator.scheduling_table = np.array(simulator.scheduling_table * 10).astype('int32')
        #simulator.index = np.zeros(len(ACS.t_NFs) - 1)

    def execute_action(self, action):
        nf_id = action.scale_nf_idx
        loc_id = action.scale_in_out_node_idx
        if action.scale_in_out_idx == 1: # scale out
            unavai_insts = self.check_available_instances(nf_id, loc_id)
            avai_insts = self.check_available_instance_by_nf_id(nf_id)
            #unavai_insts = list(set(self.nfs[nf_id])-set(avai_insts))
            ##log.logger.debug('scaling decision: nf-%d, before n_inst = %d' % (nf_id, len(avai_insts)))
            if len(avai_insts) >= ACS.n_max_inst:
                log.logger.debug('maximum number of instances already for nf-%d in node-%d' % (nf_id, loc_id))
                metrics.value[-1].punish_reward += 1
                pass
            else:
                if len(unavai_insts) == 0:
                    log.logger.debug('no available nf instances for nf-%d in node-%d' % (nf_id, loc_id))
                    pass
                else:
                    idx = random.randint(0, len(unavai_insts)-1)
                    #self.nodes[loc_id].nfs[nf_id][unavai_insts[idx].inst_id].is_alive = True
                    #self.nodes[loc_id].nfs[nf_id][unavai_insts[idx].inst_id].is_alive_event.succeed()
                    self.add_new_inst(unavai_insts[idx])
                    ##log.logger.debug('added nf-%d-%d-%d' % (unavai_insts[idx].loc_id,unavai_insts[idx].id, unavai_insts[idx].inst_id))
            vertical_scaling = action.scale_up_dn
            for i, inst in enumerate(avai_insts):
                node_res_remain = self.nodes[inst.loc_id].get_remain_resources(inst.id, inst.inst_id)
                if inst.allocated_resource*(1+vertical_scaling[i]) > node_res_remain:
                    log.logger.debug('exceed maximum resources for node-%d' % (inst.loc_id))
                    continue
                inst.allocated_resource = inst.allocated_resource * (1+vertical_scaling[i])
        if action.scale_in_out_idx == 0:
            ##log.logger.debug('scaling decision: nf-%d: maintaining ... scale in/out' % (nf_id))
            pass
        if action.scale_in_out_idx == -1:
            avai_insts = self.check_available_instance_by_nf_id(nf_id)
            ##log.logger.debug('deleting nf-%d, before n_inst = %d' % (nf_id, len(avai_insts)))
            if len(avai_insts) == 1:
                log.logger.debug('pennish this action since at least one instance should be available for services')
                metrics.value[-1].punish_reward += 1
                pass
            else:
                unavai_insts = self.check_available_instances(nf_id, loc_id)
                if len(unavai_insts) == ACS.n_max_inst:
                    log.logger.debug('no available nf instances for nf-%d in node-%d' % (nf_id, loc_id))
                    metrics.value[-1].punish_reward += 1
                    pass
                else:
                    nf_instances = list(set(self.nodes[loc_id].nfs[nf_id])-set(unavai_insts))
                    nf_instances.sort(key=lambda NF:len(NF.message_queue.items), reverse=False)
                    ##log.logger.debug('deleting an instance nf-%d-%d-%d' % (nf_instances[0].loc_id, nf_instances[0].id, nf_instances[0].inst_id))
                    self.delete_one_inst(nf_instances[0])
            vertical_scaling = action.scale_up_dn
            for i, inst in enumerate(avai_insts):
                if inst.allocated_resource * (1-vertical_scaling[i]) < NF_CPU:
                    log.logger.debug('lower than minimum resource of nf-%d-%d-%d, punish' % (inst.loc_id, inst.id, inst.inst_id))
                    continue
                inst.allocated_resource = inst.allocated_resource * (1 - vertical_scaling[i])

        schduling_table = numpy.array(action.scheduling).reshape((ACS.n_max_inst, ACS.n_max_inst)).astype('float')
        src_insts = self.check_available_instance_by_nf_id(action.scheduling_src_nf_idx)
        dst_insts = self.check_available_instance_by_nf_id(action.scheduling_dst_nf_idx)
        if action.scheduling_dst_nf_idx not in ACS.app[action.scheduling_src_nf_idx]:
            log.logger.debug('nf-%d is not succssor of nf-%d' % (action.scheduling_dst_nf_idx, action.scheduling_src_nf_idx))
            metrics.value[-1].punish_reward += 1
            return
        for i, s_inst in enumerate(src_insts):
            ##log.logger.debug('sum_scheduling: %d' % (numpy.sum(schduling_table[i,:len(dst_insts)])))
            if numpy.sum(schduling_table[i,:len(dst_insts)]) == 0:
                schduling_table[i,:len(dst_insts)] = numpy.ones(len(dst_insts))
            else:
                schduling_table[i, :len(dst_insts)][schduling_table[i,:len(dst_insts)]==0] = 0.0001
                schduling_table[i,:len(dst_insts)] = schduling_table[i,:len(dst_insts)] / numpy.sum(schduling_table[i,:len(dst_insts)])
            ##log.logger.debug('schedulingtable = %s' % (str(schduling_table[i,:len(dst_insts)].tolist())))
            x_0 = ACS.n_msgs_one_round / (1+1/schduling_table[i,0])
            for j, d_inst in enumerate(dst_insts):
                ##log.logger.debug('i,j = %d, %d， len(weight_load) = %d' % (i,j, len(s_inst.weight_load[action.scheduling_dst_nf_idx])))
                s_inst.weight_load[action.scheduling_dst_nf_idx][j] = x_0 * (schduling_table[i,j]/schduling_table[i,0])

    def execute_action_amf(self, action):
        if action == 0: # minus one
            unavai_insts = self.check_available_instances(1, 0)
            n_inst = ACS.n_max_inst - len(unavai_insts)
            if n_inst - 1 <= 0:
                metrics.value[-1].punish_reward += 0.5
            else:
                nf_instances = list(set(self.nodes[0].nfs[1]) - set(unavai_insts))
                nf_instances.sort(key=lambda NF: len(NF.message_queue.items), reverse=False)
                nf_instances[0].is_alive = False
                nf_instances[0].is_alive_event = self.sim_env.event()
                nf_instances[0].allocated_resource = NF_CPU
                # log.logger.debug('nf-%d-%d-%d closed' % (nf_instances[0].loc_id, nf_instances[0].id, nf_instances[0].inst_id))
                avai_insts = self.check_available_instance_by_nf_id(nf_instances[0].id)
                self.sim_env.process(self.moving_flows(nf_instances[0], avai_insts))
        if action == 1: # maintain
            pass
        if action == 2: # add one
            unavai_insts = self.check_available_instances(1, 0)
            n_inst = ACS.n_max_inst - len(unavai_insts)
            if n_inst + 1 >= ACS.n_max_inst:
                metrics.value[-1].punish_reward += 0.5
            else:
                idx = np.random.randint(0, len(unavai_insts))
                unavai_insts[idx].is_alive = True
                unavai_insts[idx].is_alive_event.succeed()

    def function_attention(self, sf, ef, df):
        idx = [2,6,7,10,13,17,18]
        fea = self.sigmod_func(sf+ef+df)
        ###log.logger.debug('feature: %s' % (str(fea)))
        #weight = numpy.random.normal(0,1, size=19)
        weight = np.ones(19)
        for i in idx:
            weight[i] = 10
        ###log.logger.debug('%f' % (numpy.matmul(fea, weight)))
        return numpy.matmul(fea, weight)

    def sigmod_func(self, fea):
        for i in range(len(fea)):
            #print(fea[i])
            fea[i] = 1/(1 + math.exp(-fea[i]))
        return numpy.array(fea)

    def leakyrelu(self, x): # leak = 0.2
        leak = 0.2
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    def obtain_state(self, ep, ts): # GAT
        for i in range(ACS.n_node):
            sum = 0
            for nfs in self.nodes[i].nfs:
                for inst in nfs:
                    if inst.is_alive == True:
                        sum += inst.allocated_resource
            self.nodes[i].remaining_resource = self.nodes[i].maximum_resource - sum
        e_i_j = [[] for _ in range(ACS.n_node)]
        features = [[] for _ in range(ACS.n_node)]
        for i in range(ACS.n_node):
            node_i = self.nodes[i]
            s_v_fea = [i/ACS.n_node, 1, node_i.remaining_resource/node_i.maximum_resource, -1, -1, -1, -1, -1]
            for j in range(len(self.nodes_delay_map[i])):
                if i == j:
                    continue
                node_j = self.nodes[j]
                d_v_fea = [j/ACS.n_node, 1, node_j.remaining_resource/node_j.maximum_resource, -1, -1, -1, -1, -1]
                e_fea_i_j = [-1, self.nodes_delay_map[i][j]/ACS.max_delay_between_nodes, len(self.msg_on_road[i][j].items)/1000]
                e_i_j[i].append(self.function_attention(s_v_fea, e_fea_i_j, d_v_fea))
                features[i].append(d_v_fea)
            for nf in node_i.nfs:
                for inst in nf:
                    if inst.is_alive == True:
                        d_v_fea = [-1, -1, -1, inst.loc_id/ACS.n_node, inst.id/(len(ACS.t_NFs)), inst.inst_id/ACS.n_max_inst, inst.allocated_resource/self.nodes[inst.loc_id].maximum_resource, len(inst.message_queue.items)/inst.max_load]
                        e_fea_i_j = [1, -1, -1]
                        e_i_j[i].append(self.function_attention(s_v_fea, e_fea_i_j, d_v_fea))
                        features[i].append(d_v_fea)
        alpha_i_j = [[] for _ in range(ACS.n_node)]
        for i in range(ACS.n_node):
            sum = 0
            for j in range(len(e_i_j[i])):
                sum += math.exp(self.leakyrelu(e_i_j[i][j]))
            for j in range(len(e_i_j[i])):
                alpha_i_j[i].append(math.exp(e_i_j[i][j])/sum)
        new_features = [None for _ in range(ACS.n_node)]
        for i in range(ACS.n_node):
            sum = numpy.zeros(len(features[0][0]))
            for j in range(len(features[i])):
                fea = self.sigmod_func(features[i][j])
                sum += fea*alpha_i_j[i][j]
            new_features[i] = sum.tolist()
        state = numpy.array(new_features).reshape((1, ACS.n_node*8))
        #state = np.clip((state[0] - np.mean(state[0]))/np.std(state[0]), -5, 5)
        #state = (np.max(state[0]) - state[0])/(np.max(state[0]) - np.min(state[0]))
        #log_d.logger.debug('[ep-ts = %d-%d] state = %s' % (ep, ts, str(state.tolist())))
        return state[0]

    def compute_reward(self):
        n_insts, n_res, n_max_res = [], [], []
        for i in range(ACS.n_node):
            sum_n_inst, sum_n_res = 0, 0
            for nf in self.nodes[i].nfs:
                for inst in nf:
                    if inst.is_alive == True:
                        sum_n_inst += 1
                        sum_n_res += inst.allocated_resource
            n_insts.append(sum_n_inst)
            #log.logger.debug('node-%d maximum_res = %f, occupied_res = %f' % (i, self.nodes[i].maximum_resource, sum_n_res))
            n_res.append(self.nodes[i].maximum_resource - sum_n_res)
            n_max_res.append(self.nodes[i].maximum_resource)
        #print('dukl1 ', n_res)
        mean_res = numpy.mean(numpy.array(n_res))
        max_mean_res = numpy.mean(numpy.array(n_max_res))

        #print('dukl2', mean_res)
        #metrics.value[-1].res_mean = mean_res/numpy.mean(numpy.array(n_max_res))
        #print('dukl3 ', numpy.mean(numpy.array(n_max_res)))

        var_res = float(np.sqrt(np.sum((numpy.array(n_res)/max_mean_res - mean_res/max_mean_res)**2)))
        #max_var_res = (max_mean_res**2)*ACS.n_node

        #max_var_res = numpy.max(var_res)*var_res.shape[0]
        #min_var_res = numpy.min(var_res)*var_res.shape[0]
        #if max_var_res == min_var_res:
        #    metrics.value[-1].res_var = 1
        #else:
        #    metrics.value[-1].res_var = (max_var_res - numpy.sum(var_res)) / (max_var_res - min_var_res)

        #mean_insts = numpy.mean(numpy.array(n_insts))
        #var_inst = (numpy.array(n_insts) - mean_insts) ** 2
        #max_var_inst = numpy.max(var_inst) * var_inst.shape[0]
        #min_var_inst = numpy.min(var_inst) * var_inst.shape[0]
        #if max_var_inst == min_var_inst:
        #    metrics.value[-1].res_distribution = 1
        #else:
        #    metrics.value[-1].res_distribution = (max_var_inst - numpy.sum(var_inst)) / (max_var_inst - min_var_inst)

        if metrics.value[-1].n_succ_reqs + metrics.value[-1].n_fail_reqs == 0:
            metrics.value[-1].qos_throughput = 0.001
        else:
            metrics.value[-1].qos_throughput = (metrics.value[-1].n_succ_reqs - metrics.value[-1].n_fail_reqs) / (metrics.value[-1].n_succ_reqs + metrics.value[-1].n_fail_reqs) + 1
            #if metrics.value[-1].qos_throughput > 1.8:
            #    metrics.value[-1].qos_throughput = 2
            if metrics.value[-1].qos_throughput < 0.001:
                metrics.value[-1].qos_throughput = 0.001

        if metrics.value[-1].n_succ_reqs == 0:
            metrics.value[-1].qos_delay = 0.001
        else:
            metrics.value[-1].average_delay = metrics.value[-1].average_delay/metrics.value[-1].n_succ_reqs
            metrics.value[-1].qos_delay = metrics.value[-1].qos_delay/metrics.value[-1].n_succ_reqs
            #if metrics.value[-1].qos_delay > 0.8:
            #    metrics.value[-1].qos_delay = 1
            if metrics.value[-1].qos_delay < 0.001:
                metrics.value[-1].qos_delay = 0.001

        metrics.value[-1].qos = metrics.value[-1].qos_throughput * metrics.value[-1].qos_delay
        metrics.value[-1].res = mean_res/max_mean_res * float(np.exp(-var_res))
        metrics.value[-1].res_mean = mean_res/max_mean_res
        metrics.value[-1].res_var = float(np.exp(-var_res))
        #print('testss ',var_res, np.exp(-var_res))


        metrics.value[-1].qos_weight = 1
        metrics.value[-1].res_weight = 1

        n_amf = 0
        for i in range(ACS.n_node):
            for nf in self.nodes[i].nfs:
                for inst in nf:
                    if inst.id == 1 and inst.is_alive == True:
                        n_amf += 1
        metrics.value[-1].n_inst = n_amf
        #metrics.value[-1].time_step_reward = metrics.value[-1].qos_weight * metrics.value[-1].qos + metrics.value[-1].res_weight * metrics.value[-1].res + 5 - metrics.value[-1].punish_reward
        #metrics.value[-1].time_step_reward = metrics.value[-1].qos_weight * metrics.value[-1].qos + metrics.value[-1].res_weight * metrics.value[-1].res
        metrics.value[-1].time_step_reward = metrics.value[-1].qos_weight * metrics.value[-1].qos + (1-n_amf/ACS.n_max_inst) - metrics.value[-1].punish_reward #- metrics.value[-1].punish_reward/(ACS.n_node*(len(ACS.t_NFs)-1))
        #metrics.value[-1].time_step_reward = metrics.value[-1].qos_weight * metrics.value[-1].qos
        #log.logger.debug('qos_tp=%f, qos_delay=%f, res_mean=%f, res_var=%f, res_dist=%f, reward=%f' % (metrics.value[-1].qos_throughput, metrics.value[-1].qos_delay,metrics.value[-1].res_mean,metrics.value[-1].res_var,metrics.value[-1].res_distribution,metrics.value[-1].time_step_reward))
        return float(metrics.value[-1].time_step_reward)

    def check_load(self):
        load = []
        n_sum_amf = 0
        for i in range(ACS.n_node):
            for nf in self.nodes[i].nfs:
                for inst in nf:
                    if inst.id == 0:
                        continue
                    if inst.id == 1:
                        if inst.is_alive == True:
                            n_sum_amf += 1
                            load.append(len(inst.message_queue.items))
                        else:
                            load.append(0)
                    else:
                        if inst.is_alive == True:
                            load.append(len(inst.message_queue.items))
                    #if inst.is_alive == True:
                        #load.append(inst.allocated_resource)
                        #if inst.id == 0:
                        #    load.append(0)
                        #    continue
                        #if len(inst.message_queue.items) == 0:
                        #    load.append(0)
                        #else:
                        #    load.append(len(inst.message_queue.items))
                        #log.logger.debug('nf-%d-%d-%d load-%d' % (inst.loc_id, inst.id, inst.inst_id, len(inst.message_queue.items)))
        for i in range(ACS.n_node):
            for j in range(ACS.n_node):
                load.append(len(self.msg_on_road[i][j].items))
        print('n_amf = ', n_sum_amf)
        print('un_processed_messages = ', np.sum(np.array(load)))
        state = np.array([100*n_sum_amf, np.sum(np.array(load))])
        state = state / np.sum(state)
        #load[load==0] = 1e-5
        #print('load = \n%s' % (str(load)))
        #if np.max(load) == np.min(load):
        #    load = np.zeros(ACS.n_node*len(ACS.t_NFs))
        #    load[load==0] = 0.0001
        #else:
        #    load = (np.max(load) - load) / (np.max(load) - np.min(load))
        return state

    def new_state(self, n_reqs, ts):
        n_sum_amf = 0
        for i in range(ACS.n_node):
            for inst in self.nodes[i].nfs[1]:
                if inst.is_alive == True:
                    n_sum_amf += 1
        state = []
        if ts == 0:
            state += [n_reqs/100, 0, 0, 0, n_sum_amf/ACS.n_max_inst]
        else:
            state += [n_reqs/100, metrics.value[-2].n_succ_reqs/metrics.value[-2].total_reqs, metrics.value[-2].n_fail_reqs/metrics.value[-2].total_reqs, metrics.value[-2].qos_delay, n_sum_amf/ACS.n_max_inst]
        state = np.array(state)
        #state = state / np.sum(state)
        return state

    def amf_scaling_state(self, newly_n_reqs):
        state = [newly_n_reqs/1000]
        for i in range(ACS.n_node):
            for inst in self.nodes[i].nfs[1]:
                if inst.is_alive == True:
                    state.append(len(inst.message_queue.items) / N_CAPA)
                else:
                    state.append(0)
        state = np.array(state)
        norm_state = (state - np.min(state)) / (np.max(state - np.min(state)))
        norm_state[norm_state==0] = 0.0001
        return norm_state



    def amf_scaling_reward(self, action, n_reqs, ts):
        log.logger.debug('computing reward ... at ts-%d' % (ts))
        if action is None:
            action = 1
        n_total_reqs = n_reqs
        for i in range(ACS.n_node):
            for j in range(ACS.n_node):
                if i == j:
                    continue
                n_total_reqs += len(self.msg_on_road[i][j].items)
        n_amf_inst = 0
        for i in range(ACS.n_node):
            for inst in self.nodes[i].nfs[1]:
                if inst.is_alive == True:
                    n_total_reqs += len(inst.message_queue.items)
                    n_amf_inst += 1
        log.logger.debug('total n_reqs = %d, n_inst = %d, action = %d' % (n_total_reqs, n_amf_inst, action))

        delta_x = 0
        if action == 2:
            if n_amf_inst == ACS.n_max_inst:
                metrics.value[-1].n_inst = n_amf_inst
                capacity = n_total_reqs / n_amf_inst
                capacity -= (10 * NF_CPU / 200)
                if capacity < 0:
                    capacity = 0
                if capacity > N_CAPA * 0.9:
                    capacity = 2*N_CAPA
                #delta_x = (0.9 - capacity / N_CAPA) / 0.9 - (n_amf_inst + 1) / ACS.n_max_inst - 0.5
                delta_x = (0.9 - capacity / N_CAPA) / 0.9 - 0.5
            elif n_amf_inst < ACS.n_max_inst:
                metrics.value[-1].n_inst = n_amf_inst + 1
                capacity = n_total_reqs / (n_amf_inst + 1)
                capacity -= (10 * NF_CPU / 200)
                if capacity < 0:
                    capacity = 0
                if capacity > N_CAPA * 0.9:
                    capacity = 2*N_CAPA
                #delta_x = (0.9 - capacity / N_CAPA) / 0.9 - (n_amf_inst + 1) / ACS.n_max_inst
                delta_x = (0.9 - capacity / N_CAPA) / 0.9
        if action == 1:
            metrics.value[-1].n_inst = n_amf_inst
            capacity = n_total_reqs / n_amf_inst
            capacity -= (10 * NF_CPU / 200)
            if capacity < 0:
                capacity = 0
            if capacity > N_CAPA * 0.9:
                capacity = 2*N_CAPA
            #delta_x = (0.9 - capacity / N_CAPA) / 0.9 - n_amf_inst / ACS.n_max_inst
            delta_x = (0.9 - capacity / N_CAPA) / 0.9
        if action == 0:
            if n_amf_inst == 1:
                metrics.value[-1].n_inst = n_amf_inst
                capacity = n_total_reqs / n_amf_inst
                capacity -= (10 * NF_CPU / 200)
                if capacity < 0:
                    capacity = 0
                if capacity > N_CAPA * 0.9:
                    capacity = 2*N_CAPA
                #delta_x = (0.9 - capacity / N_CAPA) / 0.9 - n_amf_inst / ACS.n_max_inst - 0.5
                delta_x = (0.9 - capacity / N_CAPA) / 0.9
            elif n_amf_inst > 1:
                metrics.value[-1].n_inst = n_amf_inst - 1
                capacity = n_total_reqs / (n_amf_inst - 1)
                capacity -= (10 * NF_CPU / 200)
                if capacity < 0:
                    capacity = 0
                if capacity > N_CAPA * 0.9:
                    capacity = 2*N_CAPA
                #delta_x = (0.9 - capacity / N_CAPA) / 0.9 - (n_amf_inst - 1) / ACS.n_max_inst
                delta_x = (0.9 - capacity / N_CAPA) / 0.9
        log.logger.debug('capacity = %f, delta_x = %f' % (capacity, delta_x))
        delta_x = (delta_x - (-1.3)) / ((N_CAPA - 1)/N_CAPA - (-1.3))
        return delta_x




class PARAMS:
    def __init__(self, nfs, nodes, msg_on_road, nodes_delay_map):
        self.nfs = nfs
        self.nodes = nodes
        self.msg_on_road = msg_on_road
        self.nodes_delay_map = nodes_delay_map
        self.inter_arr_mean = 5

def test_action():
    action = ACTIONS()
    action.scale_in_out_idx = 0
    action.scale_nf_idx = 1
    action.scale_in_out_node_idx = 1
    action.scale_up_dn = np.ones(ACS.n_max_inst)
    action.scheduling_src_nf_idx = 2
    action.scheduling_dst_nf_idx = 5
    action.scheduling = np.ones(ACS.n_max_inst*ACS.n_max_inst)#np.random.random(ACS.n_max_inst * ACS.n_max_inst)
    return action



def run_no_delay_scenario(n_ep):
    metrics.episode_reward.clear()
    agent = PPO()#MADDPG_NEW() ##MADDPG()#HADC()###TSAND()
    for ep in range(n_ep):
        metrics.episode_reward.append([0, 0])
        log.logger.debug('episode %d\n\n\n' % (ep))
        simEnv = simpy.Environment()
        env = ENV_SIMPY(simEnv)
        params = PARAMS(env.nfs, env.nodes, env.msg_on_road, env.nodes_delay_map)
        simulator = FlowSimulator(simEnv, params, env)
        simulator.start()
        episode_reward = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        #s = env.obtain_state(ep, 0)
        s = env.check_load()
        #print(s)
        metrics.value.clear()
        x_ = np.arange(0, 6.48, 0.01)
        sin_x = np.sin(x_) * 100 + 100
        for ts in range(EP_LEN):
            log.logger.debug('time step: %d' % (ts))
            if sin_x[ts + 2] < 1e-5:
                params.inter_arr_mean = 10
            else:
                params.inter_arr_mean = 10/sin_x[ts + 2]

            metrics.value.append(VALUE())
            metrics.value[-1].n_ts = ts
            metrics.value[-1].n_episode = ep
            #print('state= ', s.tolist())
            buffer_s.append(s)

            #action = test_action()
            action = agent.choose_actions(s)
            buffer_a.append(action)



            #if sum(action.scale_up_dn) > 0:
            #    action.scale_up_dn = [a / sum(action.scale_up_dn) for a in action.scale_up_dn]
            #print('action_vertical ', action.scale_up_dn)

            #log.logger.debug('action_scale_in_out_idx = %d' % (action.scale_in_out_idx))
            #log.logger.debug('action_scale_nf_idx = %d' % (action.scale_nf_idx))

            #env.execute_action(action)
            #env.execute_new_action(action, simulator)
            simulator.scheduling_table = action.sch
            simulator.scheduling_table = np.array(simulator.scheduling_table * 10).astype('int32')
            simulator.index = np.zeros(len(ACS.t_NFs) - 1)

            simEnv.run(until=10 * (ts + 1))

            #s_ = env.obtain_state(ep, ts + 1)
            s_ = env.check_load()
            s = s_

            reward = env.compute_reward()
            buffer_r.append(reward)
            log.logger.debug('episode-%d time_step-%d, reward = %f\n' % (ep,ts, reward))
            episode_reward += reward
            metrics.value[-1].episode_reward = episode_reward

            #agent.memorize(s, action, reward, s_)
            #agent.no_delay_memory(s, action, reward, s_)

            if (ts + 1) % BATCH == 0 or ts == EP_LEN - 1:
                v_s_ = agent.critic.get_v(s_)
                #print('s_: value = %f' % (v_s_))
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                #print('discounted_r: ', discounted_r)
                bs, br = np.vstack(buffer_s), np.array(discounted_r)[:,np.newaxis]
                agent.update(bs, buffer_a, br)
                buffer_s, buffer_a, buffer_r = [], [], []
        metrics.write_to_xlsx_time_step(ep)
        metrics.episode_reward[-1] = [ep, episode_reward]
        log.logger.debug('episode_reward-%d, reward = %f' % (ep, episode_reward))
    metrics.write_to_xlsx_episode()

def run_delayed_scenarios(n_ep, n_delay):
    metrics.episode_reward.clear()
    #agent = MADDPG_NEW()#DDPG(ACS.n_node*8, ACS.actors[1][0])
    #agent = DDPG(alpha=0.0003, beta=0.0003, state_dim=ACS.n_node*8, action_dim=ACS.n_node*(len(ACS.t_NFs)-1)*ACS.n_max_inst,
                 #actor_fc1_dim=400, actor_fc2_dim=300, critic_fc1_dim=400, critic_fc2_dim=300, ckpt_dir='./', batch_size=256)
    #create_directory('./', sub_paths=['Actor', 'Target_actor', 'Critic', 'Target_critic'])
    agent = Agent()
    obs_on_road = []
    reward_history = []
    for ep in range(n_ep):
        agent.reset()
        metrics.value.clear()
        metrics.episode_reward.append([0, 0])
        ##log.logger.debug('episode %d\n\n\n' % (ep))
        simEnv = simpy.Environment()
        env = ENV_SIMPY(simEnv)
        params = PARAMS(env.nfs, env.nodes, env.msg_on_road, env.nodes_delay_map)
        simulator = FlowSimulator(simEnv, params, env)
        simulator.start()
        buffer_s, buffer_a, buffer_r = [], [], []
        x_ = np.arange(0, 6.48, 0.01)
        sin_x = np.sin(x_) * 500 + 500
        episode_reward = 0
        #is_collect_obs = True
        last_reward = None
        last_action = simulator.scheduling_table
        last_states = None
        for ts in range(EP_LEN):
            if sin_x[ts + 2] < 1e-5:
                params.inter_arr_mean = 10
            else:
                params.inter_arr_mean = 10/sin_x[ts + 2]
            log.logger.debug('time step: %d' % (ts))
            metrics.value.append(VALUE())
            metrics.value[-1].n_ts = ts
            metrics.value[-1].n_episode = ep
            s = env.amf_scaling_state(10/params.inter_arr_mean)
            #s = env.obtain_state(ep, ts)
            #s = env.new_state(10/params.inter_arr_mean, ts)
            #s = env.check_load()
            #print(s.shape)
            print('s[%d] = \n%s' % (ts, str(s.tolist())))
            #if last_states is not None:
            #    print('state_var = ', np.sqrt(np.sum((s-last_states)**2)))
            #last_states = s
            #buffer_s.append(s)
            #if ts == 0:
            #    reward = None
            #else:
                #reward = env.compute_reward()
                #episode_reward += reward
                #metrics.value[-1].episode_reward = episode_reward
                #log.logger.debug('episode-%d time_step-%d, reward = %f\n' % (ep, ts, reward))
                #buffer_r.append(reward)
            #if is_collect_obs:
            obs_on_road.append([ts, 0, s, last_reward]) #　ｎ_delay = 100
            #obs_on_road.append([ts, float(np.random.randint(n_delay-TIME_PER_STEP, n_delay)) / TIME_PER_STEP, s, last_reward])  # ｎ_delay = 100
            action = agent.choose_action_with_delayed_obs(obs_on_road, ts)
            log.logger.debug('action = %d' % (action))
            #action = None
            reward = env.amf_scaling_reward(action, 10 / params.inter_arr_mean, ts)
            if action is not None:
                env.execute_action_amf(action)
                #pass
                #simulator.scheduling_table = action.sch
                #simulator.scheduling_table = np.array(simulator.scheduling_table * 10).astype('int32')
                #simulator.index = np.zeros(len(ACS.t_NFs) - 1)
                #print('a[%d] = \n%s' % (ts, str(simulator.scheduling_table.tolist())))
                #print('action_var = ', np.sqrt(np.sum((last_action - simulator.scheduling_table)**2)))
                #last_action = np.copy(simulator.scheduling_table)
                #buffer_a.append(action)
                #is_collect_obs = True
                #print(action.h_s.tolist())
                #env.execute_new_action(action, simulator)
                #print(action.v_s.tolist())
            #else:
                #is_collect_obs = False
            #env.check_load()
            #print('before load[%d] = \n%s' % (ts, str(np.array(env.check_load()).reshape((4,10)))))

            simEnv.run(until=10 * (ts + 1))
            #reward -= metrics.value[-1].punish_reward
            metrics.value[-1].time_step_reward = reward

            #print('after load[%d] = \n%s' % (ts, str(np.array(env.check_load()))))
            #reward = env.compute_reward()

            episode_reward += metrics.value[-1].time_step_reward
            last_reward = metrics.value[-1].time_step_reward
            metrics.value[-1].episode_reward = episode_reward
            log.logger.debug('episode-%d time_step-%d, raw_reward = %f, reward = %f\n' % (ep, ts, reward, metrics.value[-1].time_step_reward))

        reward_history.append(episode_reward)
        #if (ep + 1) % 200 == 0:
        #    agent.save_models(ep + 1)

        metrics.write_to_xlsx_time_step(ep)
        metrics.episode_reward[-1] = [ep, episode_reward]
        log.logger.debug('episode_reward-%d, reward = %f' % (ep, episode_reward))
    metrics.write_to_xlsx_episode()
    #episodes = [i+1 for i in range(EP_MAX)]
    #plot_learning_curve(episodes, reward_history, title='Episode Reward', ylabel='reward', figure_file='../figures/reward.png')

if __name__ == '__main__':
    #run_no_delay_scenario(EP_MAX)
    run_delayed_scenarios(EP_MAX, 100)