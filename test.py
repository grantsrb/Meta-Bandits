import numpy as np

def get_actions(self, pis):
    """
    pis - ndarray of action probabilities shape (batch_size, n_bandits)
    """
    cumsums = np.zeros((pis.shape[0],))
    randoms = np.random.random((pis.shape[0],))
    actions = np.zeros(pis.shape)
    for i in range(pis.shape[1]):
        cumsums += pis[:,i]
        actions[:,i] = (randoms < cumsums)
        randoms += actions[:,i]
    return actions.astype(np.int32)
                                                                                                        
if __name__ == "__main__":
    ps = np.array([[.2,.8], [.1,.9]])
    summer = np.zeros_like(ps)
    for j in range(10):
        for i in range(1000):
            actions = get_actions(None, ps)
            summer += actions
    avg = summer/10
    print(avg)
