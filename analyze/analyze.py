import re, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

data_file = "no-delay-dqn-0824-linux"
outFile = open(data_file+'.log', "w")

color = ['red', 'green', 'blue']
pcolor= ['paleturquoise', 'peachpuff', 'y']
episode_reward = []
np_reward_mean = []
np_reward_var  = []


with open('../logs/debug.log') as f:
    for line in f:
        ret = re.findall(r".*Episode-\d* reward: (\d*\.\d*)", line)
        if len(ret) > 0:
            print(ret[0])
            episode_reward.append(float(ret[0]))
            outFile.write(line)
outFile.close()

fig, ax = plt.subplots()
ax.plot(episode_reward, color=color[0])
plt.savefig(data_file+'.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
plt.show()

aveEP = 4
if len(episode_reward) % aveEP == 0:
    tmp = np.array(episode_reward)
else:
    tmp = np.array(episode_reward[:-(len(episode_reward)%aveEP)])
tmp = tmp.reshape(int(len(episode_reward)/aveEP), aveEP)
np_reward_mean = np.mean(tmp, 1)
np_reward_var  = 2 * sem(tmp, axis=1, ddof=0)
x_axis = [j for j in range(np_reward_mean.size)]

fig, ax = plt.subplots()
ax.plot(x_axis, np_reward_mean, color=color[0])
ax.fill_between(x_axis, np_reward_mean - np_reward_var, np_reward_mean + np_reward_var, color=pcolor[0])
plt.savefig(data_file+'-average-'+str(aveEP)+'.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
plt.show()