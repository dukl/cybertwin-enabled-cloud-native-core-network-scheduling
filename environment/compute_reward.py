import utils.global_parameters as GP
def compute_minor_reward(is_valid, index, reqs, i):
    if is_valid is True:
        return 0
    else:
        sum = 0
        for req in reqs:
            sum += req
        return -(index+1)*reqs[i]/(sum/len(GP.msc))