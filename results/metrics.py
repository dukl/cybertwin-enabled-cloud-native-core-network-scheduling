import xlwt, xlrd
from xlutils.copy import copy
import datetime
import os

now_time = datetime.datetime.now()

class VALUE:
    def __init__(self):
        self.n_episode = 0
        self.n_ts = 0
        self.total_reqs = 0
        self.n_succ_reqs = 0
        self.n_fail_reqs = 0
        self.qos_throughput = 0
        self.average_delay = 0
        self.qos_delay = 0
        self.qos = 0
        self.qos_weight = 0
        self.res_mean = 0
        self.res_var = 0
        self.res_distribution = 0
        self.res = 0
        self.res_weight = 0
        self.time_step_reward = 0
        self.episode_reward = 0
        self.action_ops = 0
        self.action_nf_select = 0
        self.actor_node_select = 0
        self.action_src_nf_select = 0
        self.action_dst_nf_select = 0
        self.punish_reward = 0
        self.n_overload = 0
        self.n_timeout = 0
        self.n_inst = 0
        #self.action_scale_up_dn = []

    def get_list_value(self):
        return [self.n_episode, self.n_ts, self.total_reqs, self.n_succ_reqs, self.n_fail_reqs, self.qos_throughput,
                self.average_delay, self.qos_delay, self.qos, self.qos_weight, self.res_mean, self.res_var, self.res_distribution, self.res, self.res_weight, self.time_step_reward, self.episode_reward,
                self.action_ops, self.action_nf_select, self.actor_node_select, self.action_src_nf_select, self.action_dst_nf_select, self.punish_reward, self.n_overload, self.n_timeout, self.n_inst]

class METRICS:
    def __init__(self):
        self.excel_path = '../results/metrics-' +str(now_time.year) + '-' + str(now_time.month) + '-' + str(now_time.day) + '-' + str(now_time.hour) +'-' + str(now_time.minute) + '/'
        self.title = ['episode', 'time_step', 'total_reqs', 'n_succ', 'n_fail', 'QoS_throughput','aver_delay','QoS_delay','QoS','QoS_weight','Res_mean','Res_var','Res_distri','Res','Res_weight',
                      'time_step_reward','episode_reward','action_ops', 'action_nf_select', 'actor_node_select', 'action_src_nf_select', 'action_dst_nf_select', 'punish_reward', 'n_overload', 'n_timeout', 'n_inst']
        self.book = None
        self.sheet = None
        self.rows = 0
        self.sheet_name = 'metrics'
        self.value = []
        self.episode_reward = []

    #def time_step_value(self):
    #    return [self.n_episode, self.n_ts, self.total_reqs, self.n_succ_reqs, self.n_fail_reqs, self.qos_throughput, self.average_delay, self.qos_delay, self.qos, self.qos_weight, self.res_mean,
    #            self.res_var, self.res_distribution, self.res, self.res_weight, self.time_step_reward, self.episode_reward]

    def write_to_xlsx_episode(self):
        if not os.path.exists(self.excel_path+'episode.xls'):
            print('create a new file')
            book = xlwt.Workbook()
            sheet = book.add_sheet('episode_reward')
            title = ['ep_id', 'episode_reward']
            for i, t in enumerate(title):
                sheet.write(0, i, t)
            id_row = 1
            for i in range(len(self.episode_reward)):
                for j, value in enumerate(self.episode_reward[i]):
                    sheet.write(id_row, j, value)
                id_row += 1
            book.save(self.excel_path+'episode.xls')

    def write_to_xlsx_time_step(self, ep):
        if not os.path.exists(self.excel_path):
            os.mkdir(self.excel_path)
        file_name = self.excel_path + 'episode-' + str(ep) + '.xls'
        if not os.path.exists(file_name):
            book = xlwt.Workbook()  # create excel
            sheet = book.add_sheet('metrics')  # create a sheet named 'metrics'
            for j, t in enumerate(self.title):
                sheet.write(0, j, t)
            rows = 1
            for i in range(len(self.value)):
                list_value = self.value[i].get_list_value()
                for idx, value in enumerate(list_value):
                    sheet.write(rows, idx, value)
                rows += 1
            book.save(file_name)
        #else:
        #    print('old excel file')
        #    self.book = xlrd.open_workbook(self.excel_path)
        #    self.sheet = self.book.sheet_by_name('metrics')
        #    self.new_book = copy(self.book)
        #    self.new_sheet = self.new_book.get_sheet(0)
        #    for i in range(self.sheet.nrows): # old data
        #        for j, value in enumerate(self.sheet.row_values(i)):
        #            self.new_sheet.write(i, j, value)
        #    self.rows = self.sheet.nrows
        #    values = self.time_step_value()
        #    for i, value in enumerate(values):
        #        self.new_sheet.write(self.rows, i, value)
        #    self.new_book.save(self.excel_path)



metrics = METRICS()
#metrics.write_to_xlsx()

