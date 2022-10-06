import xlrd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem

data = ['metrics-2022-10-4-14-58', 'metrics-2022-10-4-22-52']
color = ['red', 'green', 'blue', 'cyan', 'yellow', 'black']
pcolor= ['paleturquoise', 'peachpuff', 'y']

n_succ = [[] for _ in range(len(data))]
n_fail = [[] for _ in range(len(data))]
n_total = [[] for _ in range(len(data))]

for idx, dt in enumerate(data):
    book = xlrd.open_workbook(dt+'/episode-100.xls')
    sheet = book.sheet_by_name('metrics')
    for i in range(sheet.nrows):
        if i == 0:
            continue
        for j, value in enumerate(sheet.row_values(i)):
            if j == 2:
                n_total[idx].append(float(value))
            if j == 3:
                n_succ[idx].append(float(value))
            if j == 4:
                n_fail[idx].append(float(value))
n_succ = np.sum(np.array(n_succ), 1)
n_fail = np.sum(np.array(n_fail), 1)
n_total = np.sum(np.array(n_total), 1)

print(n_total)
print(n_succ)
print(n_fail)
print(n_succ/n_total)
print((n_succ - n_fail)/(n_succ + n_fail))