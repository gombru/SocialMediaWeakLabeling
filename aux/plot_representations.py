import numpy as np
from pylab import zeros, arange, subplots, plt, savefig

path = '../../../datasets/SocialMedia/word2vec_mean_gt/val_InstaCities1M.txt'
file = open(path, "r")
num_topics = 400

print "Loading data ..."
print path

for line in file:
    d = line.split(',')
    regression_values = np.zeros(num_topics)
    for t in range(0, num_topics):
        regression_values[t] = d[t + 1]
    break

regression_values = np.log(regression_values)

print "Max: " + str(regression_values.max())
print "Mmin: " + str(regression_values.min())
print "Mean: " + str(np.mean(regression_values))
print "Sum: " + str(sum(regression_values))



fig = plt.figure()
ax1 = plt.subplot(111)
ax1.set_xlim([0,num_topics])
ax1.set_ylim([0,regression_values.max()])

it_axes = (arange(num_topics) * 1)

ax1.plot(it_axes, regression_values, linestyle=':', color='b')

plt.show()