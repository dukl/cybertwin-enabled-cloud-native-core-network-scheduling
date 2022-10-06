from utils.logger import log
import utils.auto_scaling_settings as ACS
import random
from results.metrics import metrics

class FLOW:
    def __init__(self, ue_id, pro_id, msg_id, creation_time):
        self.ue_id = ue_id
        self.pro_id = pro_id
        self.msg_id = msg_id
        self.index = 0
        self.creation_time = creation_time
        self.in_msg_on_road_time = 0
        self.nf_inst_id = 0

class UE:
    def __init__(self, ue_id):
        self.ue_id = ue_id
        self.pros = [0 for _ in range(len(ACS.procedure))]

class FlowSimulator:
    def __init__(self, env, params, g_env):
        self.env = env
        self.g_env = g_env
        self.total_flow_count = 0
        self.env_nfs = params.nfs
        self.params = params
        self.ues = [None for _ in range(ACS.n_max_ues)]

    def start(self):
        ##log.logger.debug('Starting simulation')
        self.env.process(self.init_arrival(0))# 0 -> RISE
        for i in range(len(self.env_nfs)):
            for j in range(len(self.env_nfs[i])):
                self.env.process(self.env_nfs[i][j].handle_flow(self.env))
        for i in range(len(self.params.nodes)):
            for j in range(len(self.params.nodes)):
                self.env.process(self.queuing(i, j))

    def queuing(self, i, j):
        #last_time = 0
        while True:
            ###log.logger.debug('msg_on_road - %d-%d' % (i, j))
            flow = yield self.params.msg_on_road[i][j].get()
            #log.logger.debug('flow-%d-%d-%d flow-in-mq-time = %f, now = %f' % (flow.ue_id, flow.pro_id, flow.msg_id, flow.in_msg_on_road_time, self.env.now))
            if self.env.now - flow.in_msg_on_road_time < self.params.nodes_delay_map[i][j]:
                #log.logger.debug('transmission ... evoke ... msg_on_road-%d-%d, delay=%f' % (i, j, self.params.nodes_delay_map[i][j] - self.env.now + flow.in_msg_on_road_time))
                yield self.env.timeout(self.params.nodes_delay_map[i][j] - self.env.now + flow.in_msg_on_road_time)
            else:
                log.logger.debug('flow-%d-%d-%d has already arrived' % (flow.ue_id, flow.pro_id, flow.msg_id))
            #if flow.in_msg_on_road_time == self.env.now:
            #    #log.logger.debug('first messages ... evoke ... msg_on_road-%d-%d, delay=%f' % (i,j,self.params.nodes_delay_map[i][j]))
            #    yield self.env.timeout(self.params.nodes_delay_map[i][j]) # transmission delay between two nodes
            #    last_time = flow.in_msg_on_road_time
            #else:
            #    #log.logger.debug('continue messages ... msg_on_road-%d-%d, delay=%f, last_time=%f' % (i, j, flow.in_msg_on_road_time - last_time, last_time))
            #    yield self.env.timeout(abs(flow.in_msg_on_road_time - last_time))
            #    last_time = flow.in_msg_on_road_time
            if len(self.env_nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id*ACS.n_node+j].message_queue.items) + 1 > ACS.n_max_load_in_nf[ACS.msg_msc[flow.msg_id][flow.index]]:
                log.logger.debug('nf-%d-%d-%d overload' % (self.env_nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id*ACS.n_node+j].loc_id, self.env_nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id*ACS.n_node+j].id, self.env_nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id*ACS.n_node+j].inst_id))
                metrics.value[-1].n_fail_reqs += 1
            else:
                if self.env_nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id*ACS.n_node+j].is_alive == True:
                    yield self.env_nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id*ACS.n_node+j].message_queue.put(flow) # which instance
                else:
                    avai_insts = self.g_env.check_available_instance_by_nf_id(ACS.msg_msc[flow.msg_id][flow.index])
                    is_avai = False
                    for inst in avai_insts: # same location
                        if inst.is_alive == True:
                            if inst.loc_id == j:
                                yield self.env_nfs[inst.id][inst.inst_id*ACS.n_node+j].message_queue.put(flow)
                                #log.logger.debug('flow-%d-%d-%d arrival nf-%d-%d-%d at time %f' % (flow.ue_id, flow.pro_id, flow.msg_id,self.env_nfs[inst.id][inst.inst_id*ACS.n_node+j].loc_id, inst.id,self.env_nfs[inst.id][inst.inst_id*ACS.n_node+j].inst_id, self.env.now))
                                is_avai = True
                                break
                    if is_avai == False:
                        for inst in avai_insts:
                            if inst.is_alive == True and inst.loc_id != j:
                                flow.nf_inst_id = inst.inst_id
                                flow.in_msg_on_road_time = self.env.now
                                yield self.g_env.msg_on_road[j][inst.loc_id].put(flow)
                                #log.logger.debug('send flow-%d-%d-%d to nf-%d-%d-%d, msg_on_road-%d-%d: leaving at time %f' % (flow.ue_id, flow.pro_id, flow.msg_id, inst.loc_id, inst.id, inst.inst_id, j, inst.loc_id, self.env.now))
                                break
                #yield self.nodes[j].nfs[ACS.msg_msc[flow.msg_id][flow.index]][flow.nf_inst_id].put(flow)
            #yield self.env.timeout(1)

    def init_arrival(self, nf_id):
        avai_rise = self.g_env.check_available_instance_by_nf_id(nf_id)
        while True:
            metrics.value[-1].total_reqs += 1
            flow = self.generate_flow(nf_id)
            ###log.logger.debug('new flow generated - %f' % (self.env.now))
            rise_id = random.randint(0, len(avai_rise)-1)
            yield avai_rise[rise_id].message_queue.put(flow)
            ##log.logger.debug('generate flow-%d-%d-%d-%d: time = %f' % (flow.ue_id, flow.pro_id, flow.msg_id, flow.index, self.env.now))
            yield self.env.timeout(self.params.inter_arr_mean)


    def generate_flow(self, nf_id):
        ue_id = random.randint(0, ACS.n_max_ues-1)
        if self.ues[ue_id] is None:
            self.ues[ue_id] = UE(ue_id)
        pro_id = random.randint(0, len(ACS.procedure) - 1)
        self.ues[ue_id].pros[pro_id] += 1
        msg_id = random.randint(0, len(ACS.procedure[pro_id])-1)
        flow = FLOW(ue_id, pro_id, ACS.procedure[pro_id][msg_id], self.env.now)
        return flow


