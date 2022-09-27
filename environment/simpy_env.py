import math
import random, simpy, numpy

import numpy as np

import utils.auto_scaling_settings as ACS
from utils.logger import log
from environment.flowsimulator import FlowSimulator
from utils.actions_definition import ACTIONS
from results.metrics import metrics, VALUE
from agent.tree_structure_agent_no_delay import TSAND

EP_MAX, EP_LEN, BATCH, GAMMA = 1000, 200, 32, 0.9

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
        self.remaining_resource = 0 # [CPU, ...]
        self.maximum_resource = 8000
        self.nfs = [[] for _ in range(len(ACS.t_NFs))] # how many instances for each service

    def get_remain_resources(self, nf_id):
        sum = 0
        for nf in self.nfs:
            for inst in nf:
                if inst.is_alive == False or inst.id == nf_id:
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
        self.allocated_resource = 1000
        self.message_queue      = aStore(sim_env, capacity=ACS.n_max_load)#simpy.Store(env=sim_env, capacity=simpy.core.Infinity) #
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

    def determize_next_nf_inst(self, next_nf_id):
        next_nf_inst_idx, n_sum = 0, 0
        log.logger.debug('[NF-%d-%d-%d] avai_insts for from nf-%d to nf-%d is %d; table=%s' % (self.loc_id, self.id, self.inst_id, self.id, next_nf_id, len(self.weight_load[next_nf_id]), str(self.weight_load[next_nf_id])))
        for i in range(len(self.weight_load[next_nf_id])):
            #log.logger.debug('[NF-%d-%d-%d] weight for %d-th inst = %d, total=%d, index=%d, sum=%d' % (self.loc_id, self.id, self.inst_id, i, self.weight_load[next_nf_id][i], sum(self.weight_load[next_nf_id]), self.n_flow_process[next_nf_id], n_sum))
            n_sum += self.weight_load[next_nf_id][i]
            if self.n_flow_process[next_nf_id] < n_sum:
                next_nf_inst_idx = i
                log.logger.debug('[NF-%d-%d-%d] choose %d-th instance' % (self.loc_id, self.id, self.inst_id, next_nf_inst_idx))
                break
        self.n_flow_process[next_nf_id] = (self.n_flow_process[next_nf_id] + 1) % sum(self.weight_load[next_nf_id])
        return self.scheduling_table[next_nf_id][next_nf_inst_idx]

    def handle_flow(self, simpy_env):
        #if self.id == 0: # RISE handling
        while True:
            yield self.is_alive_event
            log.logger.debug('[NF-%d-%d-%d][res = %f] handle_flow in nf-%d-%d-%d: time = %f' % (self.loc_id, self.id, self.inst_id, self.allocated_resource, self.loc_id, self.id, self.inst_id, self.sim_env.now))
            flow = yield self.message_queue.get()
            log.logger.debug('nf-%d-%d-%d is handling flow-%d-%d-%d-%d: time = %f' % (self.loc_id, self.id, self.inst_id, flow.ue_id, flow.pro_id, flow.msg_id, flow.index, self.sim_env.now))
            if ACS.flow_ttl[flow.msg_id] < (self.sim_env.now - flow.creation_time):
                log.logger.debug('flow-%d-%d-%d-%d timeout' % (flow.ue_id, flow.pro_id, flow.msg_id, flow.index))
                metrics.value[-1].n_fail_reqs += 1
                continue
            flow.index += 1
            if flow.index >= len(ACS.msg_msc[flow.msg_id]):
                yield self.sim_env.timeout(ACS.computing_demand[flow.msg_id][flow.index-1]/self.allocated_resource)  # processing time
                metrics.value[-1].n_succ_reqs += 1
                metrics.value[-1].average_delay += (self.sim_env.now - flow.creation_time)
                metrics.value[-1].qos_delay += (1 - (self.sim_env.now - flow.creation_time)/ACS.flow_ttl[flow.msg_id])
                log.logger.debug('flow-%d-%d-%d has been processed successfully execution time = [%f-%f]' % (flow.ue_id, flow.pro_id, flow.msg_id, flow.creation_time, self.sim_env.now))
                continue
            src_nf_inst_loc = self.loc_id
            next_nf = ACS.msg_msc[flow.msg_id][flow.index]

            next_nf_inst = self.determize_next_nf_inst(next_nf)

            #self.n_flow_process[next_nf] = (self.n_flow_process[next_nf]+1) % sum(self.weight_load[next_nf])
            #next_nf_inst = (self.inst_index[next_nf] + 1) % len(self.env.nfs[next_nf]) # round robin (RR)
            #dst_nf_inst_loc = self.env.nfs[next_nf][next_nf_inst].loc_id # which instance
            dst_nf_inst_loc = next_nf_inst.loc_id
            flow.nf_inst_id = next_nf_inst.inst_id
            #self.inst_index[next_nf] = next_nf_inst
            yield self.sim_env.timeout(ACS.computing_demand[flow.msg_id][flow.index - 1] / self.allocated_resource)  # processing time
            #if self.id == 0:
            #    yield self.sim_env.timeout(ACS.computing_demand[flow.msg_id][flow.index-1]/self.allocated_resource)  # processing time
            #else:
            #    yield self.sim_env.timeout(ACS.computing_demand[flow.msg_id][flow.index-1]/self.allocated_resource)  # processing time
            log.logger.debug('send flow-%d-%d-%d from nf-%d-%d-%d to nf-%d-%d-%d, msg_on_road-%d-%d: leaving at time %f' % (flow.ue_id, flow.pro_id, flow.msg_id, src_nf_inst_loc, self.id, self.inst_id, dst_nf_inst_loc, next_nf, next_nf_inst.inst_id, src_nf_inst_loc, dst_nf_inst_loc, self.sim_env.now))
            flow.in_msg_on_road_time = self.sim_env.now
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
                    self.nodes_delay_map[i][j] = random.random()*ACS.max_delay_between_nodes
        log.logger.debug('delays = \n%s' % (str(numpy.array(self.nodes_delay_map))))
        for i in range(len(ACS.t_NFs)):
            for j in range(ACS.n_max_inst):
                #loc_id = random.randint(0,ACS.n_node-1)
                for loc_id in range(ACS.n_node):
                    nf = NF(self.sim_env, self, ACS.t_NFs[i], j, loc_id)
                    self.nfs[i].append(nf)
                    self.nodes[loc_id].nfs[i].append(nf)
                    #self.nfs[i][-1].is_alive = True
                    #self.nfs[i][-1].is_alive_event.succeed()
            inst_id = random.randint(0, ACS.n_max_inst-1)
            loc_id  = random.randint(0, ACS.n_node - 1)
            self.nfs[i][inst_id*ACS.n_node + loc_id].is_alive_event.succeed()
            self.nfs[i][inst_id*ACS.n_node + loc_id].is_alive = True
            log.logger.debug('nf-%d-%d-%d evoked' % (self.nfs[i][inst_id*ACS.n_node + loc_id].loc_id, self.nfs[i][inst_id*ACS.n_node + loc_id].id, self.nfs[i][inst_id*ACS.n_node + loc_id].inst_id))
        self.update_scheculing_table()

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
            log.logger.debug('Punnish? .. adding inst failed, because exceed maximum resource of node-%d (%f)' % (added_inst.loc_id, remain_resource_in_node))
            return
        added_inst.is_alive = True
        added_inst.is_alive_event.succeed()
        for t in ACS.t_NFs:
            if t == added_inst.id:
                continue
            for inst in self.nfs[t]:
                if inst.is_alive == False:
                    continue
                log.logger.debug('nf-%d-%d-%d is adding nf-%d-%d-%d' % (inst.loc_id, inst.id, inst.inst_id, added_inst.loc_id, added_inst.id, added_inst.inst_id))
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
        log.logger.debug('nf-%d-%d-%d evoked' % (added_inst.loc_id, added_inst.id, added_inst.inst_id))
    def delete_one_inst(self, delated_inst):
        delated_inst.is_alive = False
        delated_inst.is_alive_event = self.sim_env.event()
        delated_inst.scheduling_table = [[] for _ in range(len(ACS.t_NFs))]
        delated_inst.weight_load = [[] for _ in range(len(ACS.t_NFs))]
        delated_inst.allocated_resource = 1000
        for t in ACS.t_NFs:
            if t == delated_inst.id:
                continue
            for inst in self.nfs[t]:
                if inst.is_alive == False:
                    continue
                log.logger.debug('nf-%d-%d-%d is deleting nf-%d-%d-%d' % (inst.loc_id, inst.id, inst.inst_id, delated_inst.loc_id, delated_inst.id, delated_inst.inst_id))
                if delated_inst.id in ACS.app[inst.id]:
                    idx = inst.scheduling_table[delated_inst.id].index(delated_inst)
                    del inst.weight_load[delated_inst.id][idx]
                    del inst.scheduling_table[delated_inst.id][idx]
        log.logger.debug('nf-%d-%d-%d closed' % (delated_inst.loc_id, delated_inst.id, delated_inst.inst_id))
        avai_insts = self.check_available_instance_by_nf_id(delated_inst.id)
        self.sim_env.process(self.moving_flows(delated_inst, avai_insts))

    def moving_flows(self, deleted_nf, same_nf):
        log.logger.debug('moving flows at time %f' % (self.sim_env.now))
        n_processed_msg = len(deleted_nf.message_queue.items)
        index = 0
        for _ in range(n_processed_msg):
            if len(deleted_nf.message_queue.items) > 0:
                flow = yield deleted_nf.message_queue.get()
                yield same_nf[index].message_queue.put(flow)
                index = (index + 1) % len(same_nf)

    def execute_action(self, action):
        nf_id = action.scale_nf_idx
        loc_id = action.scale_in_out_node_idx
        if action.scale_in_out_idx == 1: # scale out
            unavai_insts = self.check_available_instances(nf_id, loc_id)
            avai_insts = self.check_available_instance_by_nf_id(nf_id)
            #unavai_insts = list(set(self.nfs[nf_id])-set(avai_insts))
            log.logger.debug('scaling decision: nf-%d, before n_inst = %d' % (nf_id, len(avai_insts)))
            if len(avai_insts) >= ACS.n_max_inst:
                log.logger.debug('maximum number of instances already for nf-%d in node-%d' % (nf_id, loc_id))
            else:
                if len(unavai_insts) == 0:
                    log.logger.debug('no available nf instances for nf-%d in node-%d' % (nf_id, loc_id))
                else:
                    idx = random.randint(0, len(unavai_insts)-1)
                    #self.nodes[loc_id].nfs[nf_id][unavai_insts[idx].inst_id].is_alive = True
                    #self.nodes[loc_id].nfs[nf_id][unavai_insts[idx].inst_id].is_alive_event.succeed()
                    self.add_new_inst(unavai_insts[idx])
                    log.logger.debug('added nf-%d-%d-%d' % (unavai_insts[idx].loc_id,unavai_insts[idx].id, unavai_insts[idx].inst_id))
        if action.scale_in_out_idx == 0:
            log.logger.debug('scaling decision: nf-%d: maintaining ... scale in/out' % (nf_id))
        if action.scale_in_out_idx == -1:
            avai_insts = self.check_available_instance_by_nf_id(nf_id)
            log.logger.debug('deleting nf-%d, before n_inst = %d' % (nf_id, len(avai_insts)))
            if len(avai_insts) == 1:
                log.logger.debug('pennish this action since at least one instance should be available for services')
            else:
                unavai_insts = self.check_available_instances(nf_id, loc_id)
                if len(unavai_insts) == ACS.n_max_inst:
                    log.logger.debug('no available nf instances for nf-%d in node-%d' % (nf_id, loc_id))
                else:
                    nf_instances = list(set(self.nodes[loc_id].nfs[nf_id])-set(unavai_insts))
                    nf_instances.sort(key=lambda NF:len(NF.message_queue.items), reverse=False)
                    log.logger.debug('deleting an instance nf-%d-%d-%d' % (nf_instances[0].loc_id, nf_instances[0].id, nf_instances[0].inst_id))
                    self.delete_one_inst(nf_instances[0])
        avai_insts = self.check_available_instance_by_nf_id(nf_id)
        #vertical_scaling = numpy.array(action.scale_up_dn).reshape((ACS.n_node, ACS.n_max_inst)).astype('float')
        vertical_scaling = action.scale_up_dn
        for i, inst in enumerate(avai_insts):
            node_res_remain = self.nodes[inst.loc_id].get_remain_resources(inst.id)
            #log.logger.debug('node-%d remains %f resource for nf-%d; %f' % (inst.loc_id, node_res_remain, inst.id, vertical_scaling[i]))
            inst.allocated_resource = node_res_remain*(vertical_scaling[i])
            if inst.allocated_resource < 100: # minimum resource for nf
                inst.allocated_resource = 100
        schduling_table = numpy.array(action.scheduling).reshape((ACS.n_max_inst, ACS.n_max_inst)).astype('float')
        src_insts = self.check_available_instance_by_nf_id(action.scheduling_src_nf_idx)
        dst_insts = self.check_available_instance_by_nf_id(action.scheduling_dst_nf_idx)
        if action.scheduling_dst_nf_idx not in ACS.app[action.scheduling_src_nf_idx]:
            log.logger.debug('nf-%d is not succssor of nf-%d' % (action.scheduling_dst_nf_idx, action.scheduling_src_nf_idx))
            return
        for i, s_inst in enumerate(src_insts):
            schduling_table[i,:len(dst_insts)] = schduling_table[i,:len(dst_insts)] / numpy.sum(schduling_table[i,:len(dst_insts)])
            #log.logger.debug('schedulingtable = %s' % (str(sum(schduling_table[i,:len(dst_insts)]))))
            x_0 = ACS.n_msgs_one_round / (1+1/schduling_table[i,0])
            for j, d_inst in enumerate(dst_insts):
                log.logger.debug('i,j = %d, %d， len(weight_load) = %d' % (i,j, len(s_inst.weight_load[action.scheduling_dst_nf_idx])))
                s_inst.weight_load[action.scheduling_dst_nf_idx][j] = x_0 * (schduling_table[i,j]/schduling_table[i,0])

    def function_attention(self, sf, ef, df):
        fea = self.sigmod_func(sf+ef+df)
        #log.logger.debug('feature: %s' % (str(fea)))
        weight = numpy.random.normal(0,1, size=19)
        #log.logger.debug('%f' % (numpy.matmul(fea, weight)))
        return numpy.matmul(fea, weight)

    def sigmod_func(self, fea):
        for i in range(len(fea)):
            fea[i] = 1/(1 + math.exp(-fea[i]))
        return numpy.array(fea)

    def leakyrelu(self, x): # leak = 0.2
        leak = 0.2
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

    def obtain_state(self): # GAT
        e_i_j = [[] for _ in range(ACS.n_node)]
        features = [[] for _ in range(ACS.n_node)]
        for i in range(ACS.n_node):
            node_i = self.nodes[i]
            s_v_fea = [i, node_i.maximum_resource, node_i.remaining_resource, -1, -1, -1, -1, -1]
            for j in range(len(self.nodes_delay_map[i])):
                if i == j:
                    continue
                node_j = self.nodes[j]
                d_v_fea = [j, node_j.maximum_resource, node_j.remaining_resource, -1, -1, -1, -1, -1]
                e_fea_i_j = [0, self.nodes_delay_map[i][j], len(self.msg_on_road[i][j].items)]
                e_i_j[i].append(self.function_attention(s_v_fea, e_fea_i_j, d_v_fea))
                features[i].append(d_v_fea)
            for nf in node_i.nfs:
                for inst in nf:
                    if inst.is_alive == True:
                        d_v_fea = [-1, -1, -1, inst.loc_id, inst.id, inst.inst_id, inst.allocated_resource, len(inst.message_queue.items)]
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
        log.logger.debug('state = %s' % (str(state)))
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
            n_res.append(self.nodes[i].maximum_resource - sum_n_res)
            n_max_res.append(self.nodes[i].maximum_resource)
        mean_res = numpy.mean(numpy.array(n_res))
        metrics.value[-1].res_mean = mean_res/numpy.mean(numpy.array(n_max_res))

        var_res = (numpy.array(n_res) - mean_res)**2
        max_var_res = numpy.max(var_res)*var_res.shape[0]
        min_var_res = numpy.min(var_res)*var_res.shape[0]
        if max_var_res == min_var_res:
            metrics.value[-1].res_var = 1
        else:
            metrics.value[-1].res_var = (max_var_res - numpy.sum(var_res)) / (max_var_res - min_var_res)

        mean_insts = numpy.mean(numpy.array(n_insts))
        var_inst = (numpy.array(n_insts) - mean_insts) ** 2
        max_var_inst = numpy.max(var_inst) * var_inst.shape[0]
        min_var_inst = numpy.min(var_inst) * var_inst.shape[0]
        if max_var_inst == min_var_inst:
            metrics.value[-1].res_distribution = 1
        else:
            metrics.value[-1].res_distribution = (max_var_inst - numpy.sum(var_inst)) / (max_var_inst - min_var_inst)

        if metrics.value[-1].n_succ_reqs + metrics.value[-1].n_fail_reqs == 0:
            metrics.value[-1].qos_throughput = 0.001
        else:
            metrics.value[-1].qos_throughput = (metrics.value[-1].n_succ_reqs - metrics.value[-1].n_fail_reqs) / (metrics.value[-1].n_succ_reqs + metrics.value[-1].n_fail_reqs) + 1
        if metrics.value[-1].qos_throughput > 1.8:
            metrics.value[-1].qos_throughput = 2
        if metrics.value[-1].qos_throughput < 0.5:
            metrics.value[-1].qos_throughput = 0.001

        if metrics.value[-1].n_succ_reqs == 0:
            metrics.value[-1].qos_delay = 0.001
        else:
            metrics.value[-1].average_delay = metrics.value[-1].average_delay/metrics.value[-1].n_succ_reqs
            metrics.value[-1].qos_delay = metrics.value[-1].qos_delay/metrics.value[-1].n_succ_reqs + 1
        if metrics.value[-1].qos_delay > 1.8:
            metrics.value[-1].qos_delay = 2
        if metrics.value[-1].qos_delay < 1.3:
            metrics.value[-1].qos_delay = 1

        metrics.value[-1].qos = metrics.value[-1].qos_throughput * metrics.value[-1].qos_delay
        metrics.value[-1].res = metrics.value[-1].res_mean * metrics.value[-1].res_var * metrics.value[-1].res_distribution

        metrics.value[-1].qos_weight = 1
        metrics.value[-1].res_weight = 1

        metrics.value[-1].time_step_reward = metrics.value[-1].qos_weight * metrics.value[-1].qos + metrics.value[-1].res_weight * metrics.value[-1].res

        return metrics.value[-1].time_step_reward

class PARAMS:
    def __init__(self, nfs, nodes, msg_on_road, nodes_delay_map):
        self.nfs = nfs
        self.nodes = nodes
        self.msg_on_road = msg_on_road
        self.nodes_delay_map = nodes_delay_map
        self.inter_arr_mean = 5

if __name__ == '__main__':
    agent = TSAND()
    for ep in range(EP_MAX):
        metrics.episode_reward.append([0, 0])
        log.logger.debug('episode %d\n\n\n' % (ep))
        simEnv = simpy.Environment()
        env = ENV_SIMPY(simEnv)
        params = PARAMS(env.nfs, env.nodes, env.msg_on_road, env.nodes_delay_map)
        simulator = FlowSimulator(simEnv, params, env)
        simulator.start()
        episode_reward = 0
        buffer_s, buffer_a, buffer_r = [], [], []
        s = env.obtain_state()
        for ts in range(EP_LEN):
            log.logger.debug('time step: %d\n' % (ts))
            metrics.value.append(VALUE())
            metrics.value[-1].n_ts = ts
            metrics.value[-1].n_episode = ep

            buffer_s.append(s)

            action = agent.choose_actions(s)
            buffer_a.append(action)

            if sum(action.scale_up_dn) > 0:
                action.scale_up_dn = [a / sum(action.scale_up_dn) for a in action.scale_up_dn]

            env.execute_action(action)

            s_ = env.obtain_state()
            s = s_

            simEnv.run(until=100*(ts+1))
            reward = env.compute_reward()
            buffer_r.append(reward)
            log.logger.debug('episode-%d time_step-%d, reward = %f' % (ep,ts, reward))
            episode_reward += reward
            metrics.value[-1].episode_reward = episode_reward

            if (ts + 1) % BATCH == 0 or ts == EP_LEN - 1:
                v_s_ = agent.critic.get_v(s_)
                discounted_r = []
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMMA * v_s_
                    discounted_r.append(v_s_)
                discounted_r.reverse()
                bs, br = np.vstack(buffer_s), np.array(discounted_r)[:,np.newaxis]
                agent.update(bs, buffer_a, br)
                buffer_s, buffer_a, buffer_r = [], [], []
        metrics.episode_reward[-1] = [ep, episode_reward]
        log.logger.debug('episode_reward-%d, reward = %f' % (ep, episode_reward))
    metrics.write_to_xlsx_episode()
    metrics.write_to_xlsx_time_step()