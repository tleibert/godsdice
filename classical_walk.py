import numpy as np

# N is the number of steps, R is the number of times we repeat the experiment with N steps
def cwalk(N, R):
    # each row corresponds to a trial R
    coin_flips = np.random.randint(0,2,(R,N))
    coin_flips[ coin_flips == 0 ] = -1 # set all 0's to -1's
    return np.sum(coin_flips,axis=1)
