import numpy as np
import pandas as pd

qsamples = pd.read_csv('grid.dat')

D = qsamples.shape[1] - 5
n = qsamples.shape[0] 
K  = ''
print('D:{}, n: {}, K: {}'.format(D,n, K))
grid = np.concatenate([qsamples['n'].values[:,np.newaxis],qsamples[['V' + str(i+1) for i in range(D)]].values], axis=1)
grid = np.concatenate([grid,np.zeros(D+1)[np.newaxis,:]], axis=0)
np.savetxt('grids/{}_{}_nopti{}'.format(n,D, K), grid)
#np.savetxt('{}_{}_nopti{}'.format(n,D, K), grid)

