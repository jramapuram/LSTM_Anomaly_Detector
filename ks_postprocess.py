import re
import sys
import numpy as np
import pandas as pd
from scipy import stats
from os import listdir, walk
from os.path import isfile, join

def kd_test(data, batch_size):
    data = np.asarray(data)
    ks_p = []
    print 'iterations to process file: ', (len(data) - batch_size)/batch_size
    for i in range(0, len(data) - batch_size, batch_size):
        # print 'iter = ', i, '| percent done = ', i/(len(data)-batch_size)
        i1, i2 = data[i : batch_size+i], data[i+batch_size : 2*batch_size + i]
        cumfreqs1, lowlim1, binsize1, extrapoints1 = stats.cumfreq(i1, numbins=batch_size)
        cumfreqs2, lowlim2, binsize2, extrapoints2 = stats.cumfreq(i2, numbins=batch_size)
        ks_p.append(stats.ks_2samp(cumfreqs1, cumfreqs2))
    return pd.DataFrame(ks_p)

# read all of the hidden layers over time
file_list = [f for f in listdir(sys.argv[1])
             if isfile(join(sys.argv[1], f)) and 'csv' in f and '(' in f]

for f in file_list:
    if 'h' in f: # Only postprocess hidden layers
        data = pd.read_csv(join(sys.argv[1], f))
        batch_size = max([int(x) for x in re.search(r'\d+, \d+', f).group(0).split(',')])
        if batch_size > 1:
            print 'running ks test on ', f
            kd_test(data, batch_size).to_csv(join(sys.argv[1]
                                                  , f.split('.csv')[0] + '_processed.csv'))
            print 'written to ', f.split('.csv')[0]+ '_processed.csv'
