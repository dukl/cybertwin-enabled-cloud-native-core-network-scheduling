import utils.auto_scaling_settings as ACS

class ACTIONS:
    def __init__(self):
        #self.scale_in_out_idx = 0  # [-1, 0, 1] scale in, maintain, scale out
        #self.scale_nf_idx = 0
        #self.scale_in_out_node_idx = 0 # [0, ACS.n_node]
        #self.scale_up_dn = [0 for _ in range(ACS.n_max_inst)] # scale up/down -> cpu percent
        #self.scheduling_src_nf_idx = 0
        #self.scheduling_dst_nf_idx = 0
        #self.scheduling = [0 for _ in range(ACS.n_max_inst*ACS.n_max_inst)]
        self.h_s = None
        self.v_s = None
        self.sch = None
