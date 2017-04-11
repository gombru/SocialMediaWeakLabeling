import numpy as np

def load_regressions_from_txt(path, num_topics):

    database = {}

    file = open(path, "r")

    print "Loading data ..."
    print path

    for line in file:
        d = line.split(',')
        regression_values = np.zeros(num_topics)
        for t in range(0,num_topics-1):
            regression_values[t] = d[t+1]
        database[d[0]] = regression_values

    return database
