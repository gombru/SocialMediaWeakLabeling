# Loads a txt with lines city/{im_id},score1,score2 into a dictionary with city/{im_id} as keys and np arrays of dim num_topics as values

import numpy as np

def load_regressions_from_txt(path, num_topics):

    database = {}

    file = open(path, "r")

    print "Loading data ..."
    print path

    for line in file:
        d = line.split(',')
        regression_values = np.zeros(num_topics)
        for t in range(0,num_topics):
            # regression_values[t-1] = d[t+1]  #-1 in regression values should not be there. So I'm skipping using topic 0 somewhere
            regression_values[t] = d[t + 1]
        database[d[0]] = regression_values

    return database
