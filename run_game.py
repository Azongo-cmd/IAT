from math import gamma
from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent
from epsilon_profile import EpsilonProfile
import logAnalysis

def main():

    #controller = KeyboardController()
    #controller = RandomAgent(game.na)
    #state = game.reset()
 
    n_episodes = 50
    max_steps = 10000
    alpha = 0.1
    eps_profile = EpsilonProfile(1.0, 0.01)
    gamma = 1.
    game = SpaceInvaders(eps_profile, gamma, alpha, display=True)
    game.learn(n_episodes, max_steps)
    """while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)"""

if __name__ == '__main__' :
    #main()
    Qlog = logAnalysis.logAnalysis("logQ.csv")
    Vlog = logAnalysis.logAnalysis("logV.csv")
    Qlog.printCurves()
    Vlog.printCurves()

