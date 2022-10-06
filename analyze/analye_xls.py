import xlrd, os, re
import matplotlib.pyplot as plt

for file in os.listdir('../results'):
    if re.search(r"10-44-episode", file) is None:
        continue
    data_path = '../results/' + file
    print(data_path)
    episode_reward = []
    book = xlrd.open_workbook(data_path)
    sheet = book.sheet_by_name('episode_reward')
    for i in range(sheet.nrows):
        if i == 0:
            continue
        value = sheet.row_values(i)
        print(value)
        episode_reward.append(float(value[1]))
    print(episode_reward)
    fig, ax = plt.subplots()
    ax.plot(episode_reward, color='red', label='episode_reward')
    plt.legend(loc=0)
    plt.xlabel('Training Episode')
    plt.ylabel('Episode Reward')
    plt.show()


