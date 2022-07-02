from time import sleep
import pygame
import random
import math
from pygame import mixer
import numpy as np
import os
from epsilon_profile import EpsilonProfile
import pandas as pd


def getURL(filename):
    return os.path.dirname(__file__) + "/" + filename

#encodes action as integer : 
#0 : gauche
#1 : droite
#2 : shoot
#3 : pass

#encodes state as np.array(np.array(pixels))

class SpaceInvaders():

    NO_INVADERS = 2
     # Nombre d'aliens  
    def __init__(self, eps_profile: EpsilonProfile, gamma: float, alpha: float, display : bool = False):
        # player
        self.display = display
        
        # nombre d'actions (left, right, fire, no_action)
        self.na = 4
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
           

        self.eps_profile = eps_profile
        self.epsilon = self.eps_profile.initial

        #   pygame
        pygame.init()

        # creating screen
        self.screen_width = 800
        self.screen_height = 600
        if self.display:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        else:
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height), flags=pygame.HIDDEN)

        # caption and icon
        pygame.display.set_caption("Welcome to Space Invaders Game by:- styles")

        # Score
        self.scoreX = 5
        self.scoreY = 5
        self.font = pygame.font.Font('freesansbold.ttf', 20)
        self.alien_size = 64
        # Initialisation des états
        self.states = list()
        self.screenX = int(self.screen_width/self.alien_size)
        self.screenY = int(self.screen_height/self.alien_size)
        print(self.screenY)
        for dx in range(0, self.screenX):
            for dy in range (-self.screenY, self.screenY):
                for d in [0,1]:
                    for i in [0,1]:
                        if i == 0:
                            state = [dx, dy, d, 'rest']
                            
                            self.states.append(state)
                        else:
                            state = [dx, dy, d, 'fire']
                            
                            self.states.append(state)

        for state in self.states:
            self.Q[str(state)] = [0,0,0,0]

         #Visualisation

        self.qvalues = pd.DataFrame(data={'episode': [], 'value': []})
        self.values = pd.DataFrame(data={'episode': [], 'value': []})

        # Game Over
        self.game_over_font = pygame.font.Font('freesansbold.ttf', 64)

        self.playerImage = pygame.image.load(getURL('data/spaceship.png'))
        self.reset()
    
    def get_player_X(self) -> int:
        return self.player_X

    def get_player_Y(self) -> int:
        return self.player_Y

    def get_indavers_X(self) -> 'List[int]':
        return self.invader_X

    def get_indavers_Y(self) -> 'List[int]':
        return self.invader_Y

    def get_bullet_X(self) -> int:
        return self.bullet_X

    def get_bullet_Y(self) -> int:
        return self.bullet_Y

    def get_bullet_state(self) -> str:
        """Projectile
        - rest = bullet is not moving
        - fire = bullet is moving
        """
        return self.bullet_state

    def full_image(self):
        return pygame.surfarray.array3d(self.screen)
    
    def getCell(self, value, size):
        return int(value/size)

    def invaderCible(self):
        k_max = 0
        for i in range(SpaceInvaders.NO_INVADERS):
            if self.get_indavers_Y()[i] > self.get_indavers_Y()[k_max]:
               k_max = i
        return k_max
        

    def get_state(self):
        """ A COMPLETER AVEC VOTRE ETAT
        Cette méthode doit renvoyer l'état du système comme vous aurez choisi de
        le représenter. Vous pouvez utiliser les accesseurs ci-dessus pour cela. 
        """
        x_distance = self.getCell(self.get_player_X(), self.screen_width) - self.getCell(self.get_indavers_X()[self.invaderCible()], self.screen_width)
        y_distance = self.getCell(self.get_player_Y(), self.screen_height) - self.getCell(self.get_indavers_Y()[self.invaderCible()], self.screen_height)
        if x_distance < 0:
            return  [abs(x_distance), y_distance, 0, self.get_bullet_state()]
        elif y_distance < 0:
            exit(0)
            #return [x_distance, abs(y_distance), 0, self.get_bullet_state()]
        else:
            return [x_distance, y_distance, 1, self.get_bullet_state()]
    
        
        #return "L'état n'est pas implémenté (SpaceInvaders.get_state)"
    def learn( self, n_episodes, max_steps):
        n_steps = np.zeros(n_episodes) + max_steps
        # Execute N episodes 
        for episode in range(n_episodes):
            # Reinitialise l'environnement
            state = self.reset()
            # Execute K steps 
            for step in range(max_steps):
                # Selectionne une action 
                action = self.select_action(state)
                # Echantillonne l'état suivant et la récompense
                next_state, reward, terminal = self.step(action)
                # Mets à jour la fonction de valeur Q
                self.updateQ(state, action, reward, next_state)
                if terminal:
                    n_steps[episode] = step + 1 
                    break

                state = next_state
            # Mets à jour la valeur du epsilon
            self.epsilon = max(self.epsilon - self.eps_profile.dec_episode / (n_episodes - 1.), self.eps_profile.final)
            # Sauvegarde et affiche les données d'apprentissage
            if n_episodes >= 0:
                print("\r#> Ep. {}/{} Value {}".format(episode, n_episodes, self.Q[str(state)][self.select_greedy_action(state)]), end =" ")
                self.save_log(state, episode)

        self.values.to_csv('logV.csv')
        self.qvalues.to_csv('logQ.csv')

    def updateQ(self, state : 'Tuple[int, int, bool, str]', action : int, reward : float, next_state : 'Tuple[int, int, bool, str]'):
        action = int(action)
        self.Q[str(state)][action] = self.Q[str(state)][action] * (1.0 - self.alpha) + self.alpha * (reward + self.gamma * np.max(self.Q[str(next_state)]))
        #raise NotImplementedError("Q-learning NotImplementedError at Function updateQ.")
    
    def select_action(self, state : 'Tuple[int, int, bool, str]'):
        """
        Cette méthode retourne une action échantilloner selon le processus d'exploration (ici epsilon-greedy).

        :param state: L'état courant
        :return: L'action 
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.na)      # random action
        else:
            return self.select_greedy_action(state)
    

    
    def select_greedy_action(self, state : 'Tuple[int, int, bool, str]'):
        """
        Cette méthode retourne l'action gourmande.

        :param state: L'état courant
        :return: L'action gourmande
        """
        mx = np.max(self.Q[str(state)])
        # greedy action with random tie break
        return np.random.choice(np.where(self.Q[str(state)] == mx)[0])
    
    def reset(self):
        """Reset the game at the initial state.
        """
        self.score_val = 0

        self.player_X = 370
        self.player_Y = 523
        self.player_Xchange = 0

        # Invader
        self.invaderImage = []
        self.invader_X = []
        self.invader_Y = []
        self.invader_Xchange = []
        self.invader_Ychange = []
        for _ in range(SpaceInvaders.NO_INVADERS):
            self.invaderImage.append(pygame.image.load(getURL('data/alien.png')))
            self.invader_X.append(random.randint(64, 737))
            self.invader_Y.append(random.randint(30, 180))
            self.invader_Xchange.append(1.2)
            self.invader_Ychange.append(50)

        # Bullet
        # rest - bullet is not moving
        # fire - bullet is moving
        self.bulletImage = pygame.image.load(getURL('data/bullet.png'))
        self.bullet_X = 0
        self.bullet_Y = 500
        self.bullet_Xchange = 0
        self.bullet_Ychange = 3
        self.bullet_state = "rest"

        if self.display:
            self.render()
    
        return self.get_state()
    
    
    def step(self, action):
        """Execute une action et renvoir l'état suivant, la récompense perçue 
        et un booléen indiquant si la partie est terminée ou non.
        """
        is_done = False
        reward = 0

        # RGB
        self.screen.fill((0, 0, 0))
        # Controling the player movement from the arrow keys
        if action == 0: # GO LEFT
            self.player_Xchange = -1.7
        if action == 1: # GO RIGHT
            self.player_Xchange = 1.7
        if action == 2: # FIRE
            self.player_Xchange = 0
            # Fixing the change of direction of bullet
            if self.bullet_state is "rest":
                self.bullet_X = self.player_X
                self.move_bullet(self.bullet_X, self.bullet_Y)
        if action == 3: # NO ACTION 
            self.player_Xchange = 0
    
        # adding the change in the player position
        self.player_X += self.player_Xchange
        for i in range(SpaceInvaders.NO_INVADERS):
            self.invader_X[i] += self.invader_Xchange[i]
    
        # bullet movement
        if self.bullet_Y <= 0:
            self.bullet_Y = 600
            self.bullet_state = "rest"
        if self.bullet_state is "fire":
            self.move_bullet(self.bullet_X, self.bullet_Y)
            self.bullet_Y -= self.bullet_Ychange
    
        # movement of the invader
        for i in range(SpaceInvaders.NO_INVADERS):
            if self.invader_Y[i] >= 450:
                if abs(self.player_X-self.invader_X[i]) < 80:
                    print('yes')
                    is_done = True
                    """for j in range(SpaceInvaders.NO_INVADERS):
                        self.invader_Y[j] = 2000"""
                    break
                
            if self.invader_X[i] >= 735 or self.invader_X[i] <= 0:
                self.invader_Xchange[i] *= -1
                self.invader_Y[i] += self.invader_Ychange[i]
            # Collision
            collision = self.isCollision(self.bullet_X, self.invader_X[i], self.bullet_Y, self.invader_Y[i])
            if collision:
                reward = 1
                self.score_val += 1
                self.bullet_Y = 600
                self.bullet_state = "rest"
                self.invader_X[i] = random.randint(64, 736)
                self.invader_Y[i] = random.randint(30, 200)
                self.invader_Xchange[i] *= -1
    
            self.move_invader(self.invader_X[i], self.invader_Y[i], i)
    
        # restricting the spaceship so that it doesn't go out of screen
        if self.player_X <= 16:
            self.player_X = 16
        elif self.player_X >= 750:
            self.player_X = 750

        self.move_player(self.player_X, self.player_Y)

        if self.display:
            self.render()
    
        return self.get_state(), reward, is_done

    def render(self):
        self.show_score(self.scoreX, self.scoreY)
        pygame.display.update()

    def move_player(self, x, y):
        self.screen.blit(self.playerImage, (x - 16, y + 10))

    def move_invader(self, x, y, i):
        self.screen.blit(self.invaderImage[i], (x, y))

    def move_bullet(self, x, y):
        self.screen.blit(self.bulletImage, (x, y))
        self.bullet_state = "fire"

    def show_score(self, x, y):
        score = self.font.render("Points: " + str(self.score_val), True, (255,255,255))
        self.screen.blit(score, (x , y ))

    def game_over(self):
        game_over_text = self.game_over_font.render("GAME OVER", True, (255,255,255))
        self.screen.blit(game_over_text, (190, 250))


    # Collision Concept
    def isCollision(self, x1, x2, y1, y2):
        distance = math.sqrt((math.pow(x1 - x2,2)) + (math.pow(y1 - y2,2)))
        return (distance <= 50)

    def save_log(self, state, episode):
        """Sauvegarde les données d'apprentissage.
        :warning: Vous n'avez pas besoin de comprendre cette méthode
        """
        
        # Construit la fonction de valeur d'état associée à Q
        V = {}
        for state in self.states:
            val = self.Q[str(state)][self.select_action(state)]
            V[str(state)] = val
        self.qvalues = self.qvalues.append({'episode': episode, 'value': self.Q[str(state)][self.select_greedy_action(state)]}, ignore_index=True)
        self.values = self.values.append({'episode': episode, 'value': V[str(state)]},ignore_index=True)
