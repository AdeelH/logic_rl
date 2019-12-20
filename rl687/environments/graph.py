import numpy as np
from typing import Tuple
from .skeleton import Environment


class Graph(Environment):

    def __init__(self, startState:int, endState:int, numStates:int, transition_map:dict, R:dict):
        self._name = "Graph"
        self._gamma = 1.
        self._startState = startState
        self._endState = endState
        self.numStates = numStates

        self._transition_map = transition_map
        self._R = R

        self.reset()

    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd

    @property
    def state(self) -> int:
        # return int(np.where(x==1)[0])
        return int(self._state)

    @property
    def gamma(self) -> float:
        return self._gamma

    def nextState(self, state: int, action: int) -> int:
        return self._transition_map(state, action)

    def R(self, state: int, action: int, nextState: int) -> float:
        return 0.

    def step(self, action: int) -> Tuple[int, float, bool]:

        self._state = self._transition_map[(self._state, action)]
        self._reward = self._R[self._state]
        self._isEnd = self._state == self._endState

        return self.state, self.reward, self.isEnd

    def reset(self) -> None:
        self._state = self._startState
        self._action = None
        self._reward = 0
        self._isEnd = False

map1 = {
    (0, 0): 1,
    (0, 1): 2,
    
    (1, 0): 3,
    (1, 1): 4,
    (2, 0): 3,
    (2, 1): 4,

    (3, 0): 5,
    (3, 1): 6,
    (4, 0): 5,
    (4, 1): 6,

    (5, 0): 7,
    (5, 1): 7,
    (6, 0): 7,
    (6, 1): 7,
}
r1 = {0: 0, 1: 0, 2: 1, 3: 1, 4: 0, 5: 0, 6: 1, 7: 0}
r1_tl = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

def graph1():
    return Graph(0, 7, 8, map1, r1)

def graph1_tl():
    return Graph(0, 7, 8, map1, r1_tl)
