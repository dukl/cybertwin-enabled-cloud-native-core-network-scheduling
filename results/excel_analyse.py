import xlrd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

data = ['metrics-2022-10-5-11-27', 'metrics-2022-10-5-12-31', 'metrics-2022-10-5-14-13']
#data = ['episode-2000-100-00001', 'episode-1000-200-00001']
color = ['red', 'green', 'blue', 'cyan', 'yellow', 'black']
pcolor= ['paleturquoise', 'peachpuff', 'y']

episode_reward = [[] for _ in range(len(data))]

for idx, dt in enumerate(data):
    book = xlrd.open_workbook(dt+'/episode.xls')
    sheet = book.sheet_by_name('episode_reward')
    for i in range(sheet.nrows):
        if i == 0:
            continue
        for j, value in enumerate(sheet.row_values(i)):
            if j == 0:
                continue
            episode_reward[idx].append(float(value))

# plot
fig, ax = plt.subplots()
for i, ep_rwd in enumerate(episode_reward):
    ax.plot(ep_rwd, color=color[i], label=data[i])
plt.legend(loc=0)
plt.xlabel('Training Episode')
plt.ylabel('Episode Reward')
plt.savefig('figures/episode-10-06.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
plt.show()

fig, ax = plt.subplots()
for i, ep_rwd in enumerate(episode_reward):
    ep_rwd = np.array(ep_rwd).reshape(int(len(ep_rwd)/10), 10)
    mean = np.mean(ep_rwd, 1)
    var  = 2 * sem(ep_rwd, axis=1, ddof=0)
    x_axis = [j for j in range(mean.size)]
    ax.plot(x_axis, mean, color=color[i], label=data[i])
    ax.fill_between(x_axis, mean - var, mean + var, color=pcolor[i])
plt.legend(loc=0)
plt.savefig('figures/episode-10-06-aver-10.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
plt.show()