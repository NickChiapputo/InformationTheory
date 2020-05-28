import numpy as np
import matplotlib.pyplot as plt

# data to plot
n_groups = 4
naive_mine = (0.259, 2.167, 3.826, 5.155)
naive_theirs = (1.8, 3.7, 5.2, 6.6)

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35
opacity = 0.8

rects1 = plt.bar(index, naive_mine, bar_width,
alpha=opacity,
color='b',
label='Naive')

rects2 = plt.bar(index + bar_width, naive_theirs, bar_width,
alpha=opacity,
color='g',
label='Paper')

plt.xlabel('Person')
plt.ylabel('Time (s)')
plt.title('Scores by person')
plt.xticks(index + bar_width, ('Delay = 0 s', 'Delay = 2 s', 'Delay = 3.5 s', 'Delay = 5 s'))
plt.legend()

plt.tight_layout()
plt.show()