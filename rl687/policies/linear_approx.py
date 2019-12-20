import numpy as np
from .skeleton import Policy
from typing import Union

class LinearApprox(Policy):
    """
    A Tabular Softmax Policy (bs)


    Parameters
    ----------
    numStates (int): the number of states the tabular softmax policy has
    numActions (int): the number of actions the tabular softmax policy has
    """

    def __init__(self, numActions: int):
        self.numActions = numActions
        self._theta = np.random.randn(5, numActions)

    def phi(self, state:np.ndarray)->np.ndarray:
        """Feature map

        Arguments:
            state {np.ndarray} -- state vector
        
        Returns:
            np.ndarray -- feature vector
        """
        x, v, theta, dtheta = state
        features = np.array([x / 3, v, np.sin(theta), theta, dtheta])
        return features

    @property
    def parameters(self)->np.ndarray:
        """
        Return the policy parameters as a numpy vector (1D array).
        This should be a vector of length |S|x|A|
        """
        return self._theta.flatten()
    
    @parameters.setter
    def parameters(self, p:np.ndarray):
        """
        Update the policy parameters. Input is a 1D numpy array of size |S|x|A|.
        """
        self._theta = p.reshape(self._theta.shape)

    def __call__(self, state:int, action=None)->Union[float, np.ndarray]:
        action_probs = self.getActionProbabilities(state)
        return action_probs[action] if action is not None else action_probs

    def samplAction(self, state:int)->int:
        """
        Samples an action to take given the state provided. 
        
        output:
            action -- the sampled action
        """
        return np.random.choice(self.numActions, p=self.getActionProbabilities(state))

    def getActionProbabilities(self, state:np.ndarray)->np.ndarray:
        """
        Compute the softmax action probabilities for the state provided. 
        
        output:
            distribution -- a 1D numpy array representing a probability 
                            distribution over the actions. The first element
                            should be the probability of taking action 0 in 
                            the state provided.
        """
        out = self.phi(state).dot(self._theta).flatten()
        exps = np.exp(out - out.max())
        return exps / exps.sum()
