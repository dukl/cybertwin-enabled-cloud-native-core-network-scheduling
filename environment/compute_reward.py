import utils.global_parameters as GP
def compute_minor_reward(is_valid, index, reqs, i, nf, m, n_threads):
    if is_valid is True:
        t = nf[0]*GP.w_ms[m]/nf[1] + 1/(GP.cpu/GP.psi_ms[m] - 1)
        t_max = GP.lamda_ms[m] * GP.w_ms[m] / 1 + 1/(GP.cpu/GP.psi_ms[m] - 1)
        t_min = nf[0] * GP.w_ms[m] / GP.ypi_max + 1/(GP.cpu/GP.psi_ms[m] - 1)
        if t_max == t_min:
            return 0
        return 1 - (t_max - t)/(t_max - t_min) + 1 - n_threads*GP.psi_ms[m]/(GP.ypi_max*GP.psi_ms[m]+GP.c_r_ms[m])
    else:
        return -(index+1)*reqs[i]/(sum(reqs)/len(GP.msc))