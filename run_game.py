from time import sleep
from game.SpaceInvaders import SpaceInvaders
from controller.keyboard import KeyboardController
from controller.random_agent import RandomAgent

def main():

    game = SpaceInvaders(display=True)
    #controller = KeyboardController()
    controller = RandomAgent(game.na)
 
    state = game.reset()
    game.learn(20, 300)
    """while True:
        action = controller.select_action(state)
        state, reward, is_done = game.step(action)
        sleep(0.0001)"""

if __name__ == '__main__' :
    main()
