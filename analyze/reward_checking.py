import re, os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

color = ['red', 'green', 'blue', 'cyan', 'yellow', 'black']
pcolor= ['paleturquoise', 'peachpuff', 'y']

succ_rate = []
Q_t = []
resource_rate = []
Q_loss = []

date = '0901'
print(os.listdir(date+'/'))

for file in os.listdir(date+'/'):
    print(file)
    if re.search(r"ddpg-01", file) is None:
        continue
    with open(date + '/' + file) as f:
        for line in f:
            #print(line)
            ret = re.findall(r".*detailed-reward: succ_rate=(\d*\.\d*), Q_t=(\d*\.\d*), total_req=\d*, resource_rate=(\d*\.\d*).*", line)
            if len(ret) > 0:
                #print(ret)
                succ_rate.append(float(ret[0][0]))
                Q_t.append(float(ret[0][1]))
                resource_rate.append(float(ret[0][2]))
            ret = re.findall(r".*Q_loss = (\d*\.\d*).*", line)
            if len(ret) > 0:
                #print(ret)
                Q_loss.append(float(ret[0]))
#fig, ax = plt.subplots()
#for i in range(3):
#    ax.plot(np.array(succ_rate)[(i*100):(100+i*100)], color=color[i])
#plt.savefig(date+'/'+date+'-succ_rate.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
#plt.show()

#fig, ax = plt.subplots()
#for i in range(3):
#    ax.plot(np.array(Q_t)[(i*100):(100+i*100)], color=color[i])
#plt.savefig(date+'/'+date+'-q_t.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
#plt.show()

#fig, ax = plt.subplots()
#for i in range(3):
#    ax.plot(np.array(resource_rate)[(i*100):(100+i*100)], color=color[i])
#plt.savefig(date+'/'+date+'-resource_rate.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
#plt.show()

data = [succ_rate, Q_t, resource_rate]
label = ['succ_rate', 'Q_t', 'res_rate']

for id_episode in [200, 300, 400]:
    fig, ax = plt.subplots()
    for rwds in data:
        ax.plot(np.array(rwds)[(id_episode*100):(100+id_episode*100)], color=color[data.index(rwds)], label=label[data.index(rwds)])
    plt.legend(loc=0)
    plt.savefig(date+'/'+date+'-episode-'+str(id_episode)+'.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
    plt.show()

fig, ax = plt.subplots()
ax.plot(Q_loss, color=color[0])
plt.savefig(date+'/'+date+'-Q_loss.png', bbox_inches='tight',dpi=fig.dpi,pad_inches=0.1)
plt.show()