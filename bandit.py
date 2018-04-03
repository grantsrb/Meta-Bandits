import numpy as np

class Bandit():
    def __init__(self, n_bandits=2, probs=None):
        if probs is None:
            probs = np.random.random(n_bandits)*2.0
            probs = self.softmax(np.array(probs,dtype=np.float32))
        self.bandits = probs
        self.n_bandits = len(self.bandits)

    def pull_lever(self, one_hot_bandit):
        """
        one_hot_bandit - one hot encoding of selected bandit index. shape = (n_bandits,)
        """
        idx = np.random.choice(self.n_bandits, p=self.bandits)
        return int(one_hot_bandit[idx])

    def softmax(self, x):
        """
        Performs softmax function on a vector x

        x - list or ndarray of dimension (length,) or (batch_size,length)
        """
        x = np.asarray(x)
        squeeze_end = False
        if len(x.shape) < 2: 
            x = x[None]
            squeeze_end = True
        maxes = np.max(x, axis=-1)[:,None]
        reduced_x = x - maxes
        exps = np.exp(reduced_x)
        denoms = np.sum(exps, axis=-1)[:,None]
        if squeeze_end:
            return (exps/denoms).squeeze()
        else:
            return exps/denoms

