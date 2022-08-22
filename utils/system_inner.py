from utils.obs_reward_action_def import OBSRWD
import results.running_value as RV

def CHECK_OBSERVATIONS(ts):
    O = []
    for ob in RV.obs_on_road:
        if ob.id + ob.obs_delay <= ts:
            O.append(ob)
            RV.obs_on_road.remove(ob)
    return O
