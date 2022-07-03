from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile
import numpy as np


class Agent():

    def __init__(self, Q):
        self.Q = Q

    def select_action(self, state):
        mx = np.max(self.Q[str(state)])
        return np.random.choice(np.where(self.Q[str(state)] == mx)[0])

    def learn(self,eps_profile, gamma, alpha, n_episodes, max_steps):
        game = SpaceInvaders(eps_profile, gamma, alpha, display=True)
        game.learn(n_episodes, max_steps)
        self.Q = game.Q
