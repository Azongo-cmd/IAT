from math import gamma
from time import sleep

from matplotlib.font_manager import json_dump
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile
import numpy as np
from controller.agent import Agent
import json


    

def main():
    n_episodes = 10
    max_steps = 100000
    alpha = 0.5
    eps_profile = EpsilonProfile(1.0, 0.01)
    gamma = 1.
    game = SpaceInvaders(eps_profile, gamma, alpha, display=True)
    state = game.reset()
    #controller = RandomAgent(game.na)
    with open('Q.txt', 'r') as j:
     AgentQ = json.loads(j.read())

    agent = Agent(AgentQ)
    while True:
        action = agent.select_action(state)
        #action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)

if __name__ == '__main__' :
    main()

