"""
========
Barchart
========

A bar plot with errorbars and height labels on individual bars
"""
import numpy as np
import matplotlib.pyplot as plt

N = 4

ind = np.arange(N)  # the x locations for the groups
width = 0.15       # the width of the bars

fig, ax = plt.subplots()

men_means = (49.3, 89.3, 3.0, 3.2)
rects1 = ax.bar(ind, men_means, width, color='r')

women_means = (44.9, 66.6, 19.3, 4.1)
rects2 = ax.bar(ind + width, women_means, width, color='y')

women_means = (1.0, 5.1, 2.6, 1.0)
rects3 = ax.bar(ind + 2 * width, women_means, width, color='b')

# add some text for labels, title and axes ticks
ax.set_ylabel('Cover rate')
ax.set_title('Cluster quality of different algorithm')
ax.set_xticks(ind + width)
ax.set_xticklabels(('wine', 'iris', 'yeast', 'red-wine'))

ax.legend((rects1[0], rects2[0], rects3[0]), ('RSLFC', 'K-Means', 'DBSCAN'))


plt.show()