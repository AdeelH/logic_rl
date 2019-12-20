import numpy as np
import numpy.random as nprand
from typing import Tuple
from .skeleton import Environment

class Cartpole(Environment):
    """
    The cart-pole environment as described in the 687 course material. This
    domain is modeled as a pole balancing on a cart. The agent must learn to
    move the cart forwards and backwards to keep the pole from falling.

    Actions: left (0) and right (1)
    Reward: 1 always

    Environment Dynamics: See the work of Florian 2007
    (Correct equations for the dynamics of the cart-pole system) for the
    observation of the correct dynamics.
    """

    def __init__(self):
        # TODO: properly define the variables below
        self._name = 'cartpole'
        self._action = None
        self._reward = 0.
        self._isEnd = False
        self._gamma = 1.
        self._actions = (-10., 10.)
        self.state_dim = 4

        # define the state # NOTE: you must use these variable names
        self._x = 0.  # horizontal position of cart
        self._v = 0.  # horizontal velocity of the cart
        self._theta = 0.  # angle of the pole
        self._dtheta = 0.  # angular velocity of the pole

        self._t = 0.0  # total time elapsed

        # dynamics
        self._g = 9.8  # gravitational acceleration (m/s^2)
        self._mp = 0.1  # pole mass
        self._mc = 1.0  # cart mass
        self._l = 0.5  # (1/2) * pole length
        self._dt = 0.02  # timestep

        self._x_lim = 3
        self._t_lim = 20
        self._theta_lim = np.pi / 10

        self.random_state_scaling = np.array([.25, 0, self._theta_lim * .75, 0])

        self.init_state = (0., 0., 0., 0.)
        self.hard_start_states = [self.init_state]

    def __str__(self):
        return f'{self._t}, {self.state}, {self.isEnd}, {self.reward}'


    @property
    def name(self) -> str:
        return self._name

    @property
    def reward(self) -> float:
        return self._reward

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def action(self) -> int:
        return self._action

    @property
    def isEnd(self) -> bool:
        return self._isEnd > 0

    @property
    def state(self) -> np.ndarray:
        return np.array((self._x, self._v, self._theta, self._dtheta), dtype=np.float64)

    @state.setter
    def state(self, val):
        self._x, self._v, self._theta, self._dtheta = val

    def nextState(self, state: np.ndarray, action: int) -> np.ndarray:
        """
        Compute the next state of the pendulum using the euler approximation to the dynamics
        """
        l, mp, mc, g, dt = self._l, self._mp, self._mc, self._g, self._dt
        F = self._actions[action]
        x, v, theta, dtheta = state

        m = mc + mp
        st, ct = np.sin(theta), np.cos(theta)
        domega = ( g*st + ct*( (-F - mp*l*(dtheta**2)*st) / m ) ) / ( l * ( 4/3 - (mp*(ct**2)) / m ) )
        dv = (F + mp*l*((dtheta**2)*st - domega*ct)) / m

        return state + self._dt * np.array((v, dv, dtheta, domega), dtype=np.float64)

    def R(self, state: np.ndarray, action: int, nextState: np.ndarray) -> float:
        return 1. #(self._gamma**self._t)

    def step(self, action: int) -> Tuple[np.ndarray, float, bool]:
        """
        takes one step in the environment and returns the next state, reward, and if it is in the terminal state
        """
        next_state = self.nextState(self.state, action)
        reward = self.R(self.state, action, next_state)

        self._t += self._dt
        self.state = next_state
        self._action = action
        self._reward = reward
        self._isEnd = self.terminal()

        return next_state, reward, self._isEnd

    def reset(self, hard_reset=False, state=None, random=True) -> None:
        """
        resets the state of the environment to the initial configuration
        """
        if state is None:
            if random:
                self.state = self.hard_start_states[nprand.choice(len(self.hard_start_states))]
            else:
                self.state = self.init_state
        else:
            self.state = state
        self._t = 0.
        self._action = None
        self._isEnd = False
        if hard_reset:
            self.init_state = (0., 0., 0., 0.)
            self.hard_start_states = [self.init_state]

    def terminal(self) -> bool:
        """
        The episode is at an end if:
            time is greater that 20 seconds
            pole falls |theta| > (pi/12.0)
            cart hits the sides |x| >= 3
        """
        return \
        self._t > self._t_lim \
        or np.abs(self._theta) > self._theta_lim \
        or np.abs(self._x) >= self._x_lim

    def generate_random_state(self):
        s = nprand.uniform(-1, 1, size=4) * self.random_state_scaling
        return s
