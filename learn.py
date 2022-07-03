from matplotlib.font_manager import json_dump
from epsilon_profile import EpsilonProfile
from controller.agent import Agent
import json

def main():
    n_episodes = 15
    max_steps = 1000000
    alpha = 0.5
    eps_profile = EpsilonProfile(1.0, 0.01)
    gamma = 1.
    agent = Agent({})
    agent.learn(eps_profile, gamma, alpha, n_episodes, max_steps)
    with open('Q.txt', 'w') as f:
        f.write(json.dumps(agent.Q))
if __name__ == '__main__' :
    main()
