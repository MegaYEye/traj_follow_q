
from collections import defaultdict
import random, math
import numpy as np
import pickle
from functools import partial
class QLearningAgent:
    def __init__(self, alpha, epsilon, discount, get_legal_actions):

        self.get_legal_actions = get_legal_actions
        #self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
        self._qvalues = defaultdict(partial(defaultdict,int))
        self.alpha = alpha
        self.epsilon = epsilon
        self.discount = discount

    def get_qvalue(self, state, action):
        return self._qvalues[state][action]

    def set_qvalue(self,state,action,value):
        self._qvalues[state][action] = value
        
    def get_str_qvalue(self,state):
        return str(self._qvalues[state])

    def get_value(self, state):

        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return 0.0
        if len(possible_actions) == 0:
            return 0.0

        value=None
        for action in possible_actions:
            val=self.get_qvalue(state,action)
            if value is None:
                value=val
            if value<val:
                value=val

        return value

    def update(self, state, action, reward, next_state):

        #agent parameters
        gamma = self.discount
        learning_rate = self.alpha

        Q= (1-learning_rate) * self.get_qvalue(state,action) + learning_rate * (reward+gamma*self.get_value(next_state))
        
        
        self.set_qvalue(state, action, Q)

    
    def get_best_action(self, state):

        possible_actions = self.get_legal_actions(state)

        #If there are no legal actions, return None
        if len(possible_actions) == 0:
            return None

        value=None
        best_action=None
#         print(possible_actions)
        for action in possible_actions:
            val=self.get_qvalue(state,action)
            if value is None:
                value=val
                best_action=action
            if value<val:
                value=val
                best_action=action

        return best_action

    def get_action(self, state):


        # Pick Action
        possible_actions = self.get_legal_actions(state)
        action = None

        if len(possible_actions) == 0:
            return None

        epsilon = self.epsilon
        
   
        if random.random()<epsilon:
            chosen_action=random.choice(possible_actions)
        else:
            chosen_action=self.get_best_action(state)        
        
        return chosen_action

    def save_param(self, path):
        with open(path, "wb") as p:
            pickle.dump(self._qvalues, p, protocol=pickle.HIGHEST_PROTOCOL)
            
    def load_param(self, path):
        try:
            with open(path, 'rb') as p:
                self._qvalues = pickle.load(p)
                # print(self._qvalues)
        except:
            print("load param failure! let's start from a empty q table!")
            #self._qvalues = defaultdict(lambda: defaultdict(lambda: 0))
            self._qvalues = defaultdict(partial(defaultdict,int))