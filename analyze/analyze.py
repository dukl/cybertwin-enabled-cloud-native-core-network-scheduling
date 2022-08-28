import re, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

color = ['red', 'green', 'blue', 'cyan', 'yellow', 'black']
pcolor= ['paleturquoise', 'peachpuff', 'y']
episode_reward = []
np_reward_mean = []
np_reward_var  = []
labels         = []

date = '0828'
print(os.listdir(date+'/'))
n_files = 0

for file in os.listdir(date+'/'):
    print(file)
    if re.search(r"log", file) is None:
        continue
    n_files += 1
    labels.append(file)
    tmp_reward = []
    with open(date + '/' + file) as f:
        for line in f:
            ret = re.findall(r".*Episode-\d* reward: (\d*\.\d*)", line)
            if len(ret) > 0:
                print(ret[0])
                tmp_reward.append(float(ret[0]))
    episode_reward.append(tmp_reward)

fig, ax = plt.subplots()
for i in range(n_files):
    ax.plot(episode_reward[i], color=color[i], label=labels[i])
plt.legend(loc=0)
plt.xlabel('Training Episode')
plt.ylabel('Episode Reward')
plt.savefig(date+'/'+date+'.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
plt.show()

#aveEP = 4
#if len(episode_reward) % aveEP == 0:
#    tmp = np.array(episode_reward)
#else:
#    tmp = np.array(episode_reward[:-(len(episode_reward)%aveEP)])
#tmp = tmp.reshape(int(len(episode_reward)/aveEP), aveEP)
#np_reward_mean = np.mean(tmp, 1)
#np_reward_var  = 2 * sem(tmp, axis=1, ddof=0)
#x_axis = [j for j in range(np_reward_mean.size)]

#fig, ax = plt.subplots()
#ax.plot(x_axis, np_reward_mean, color=color[0])
#ax.fill_between(x_axis, np_reward_mean - np_reward_var, np_reward_mean + np_reward_var, color=pcolor[0])
#plt.savefig(data_file+'-average-'+str(aveEP)+'.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
#plt.show()