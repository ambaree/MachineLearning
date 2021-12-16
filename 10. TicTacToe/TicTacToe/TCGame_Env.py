#from gym import spaces
import numpy as np
import random
from itertools import groupby
from itertools import product


class TicTacToe():

    def __init__(self):
        """initialise the board"""
        
        # initialise state as an array
        self.state = [np.nan for _ in range(9)]  # initialises the board position, can initialise to an array or matrix
        # all possible numbers
        self.all_possible_numbers = [i for i in range(1, len(self.state) + 1)] # , can initialise to an array or matrix

        self.reset()

    # Tic-Tac-Toe board
    # 0, 1, 2
    # 3, 4, 5
    # 6, 7, 8

    def is_winning(self, curr_state):
        """Takes state as an input and returns whether any row, column or diagonal has winning sum
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan]
        Output = False"""
        
        # Game is won when 
            # 1. sum of rows is 15 or
            # 2. sum of columns is 15 or
            # 3. sum of diagnals is 15
            
        if (
            # check ROWS. cells 0,1,2 ; 3,4,5 ; 6,7,8
            sum(curr_state[0:3:1])==15 or sum(curr_state[3:6:1])==15 or sum(curr_state[6:9:1])==15 or  
            # check COLUMNS. cells 0,3,6 ; 1,4,7 ; 2,5,8
            sum(curr_state[0:9:3])==15 or sum(curr_state[1:9:3])==15 or sum(curr_state[2:9:3])==15 or  
            # check DIAGNOLS. cells 0,4,8 ; 2,4,6
            sum(curr_state[0:9:4])==15 or sum(curr_state[2:7:2])==15):                                 
            return True
        else:
            return False
 

    def is_terminal(self, curr_state):
        # Terminal state could be winning state or when the board is filled up

        if self.is_winning(curr_state) == True:
            return True, 'Win'

        elif len(self.allowed_positions(curr_state)) ==0:
            return True, 'Tie'

        else:
            return False, 'Resume'


    def allowed_positions(self, curr_state):
        """Takes state as an input and returns all indexes that are blank"""
        return [i for i, val in enumerate(curr_state) if np.isnan(val)]


    def allowed_values(self, curr_state):
        """Takes the current state as input and returns all possible (unused) values that can be placed on the board"""

        used_values = [val for val in curr_state if not np.isnan(val)]
        agent_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 !=0]
        env_values = [val for val in self.all_possible_numbers if val not in used_values and val % 2 ==0]

        return (agent_values, env_values)


    def action_space(self, curr_state):
        """Takes the current state as input and returns all possible actions, i.e, all combinations of allowed positions and allowed values"""

        agent_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[0])
        env_actions = product(self.allowed_positions(curr_state), self.allowed_values(curr_state)[1])
        return (agent_actions, env_actions)



    def state_transition(self, curr_state, curr_action):
        """Takes current state and action and returns the board position just after agent's move.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = [1, 2, 3, 4, nan, nan, nan, 9, nan]
        """
        # create next_state which is same as curr_state
        next_state = [i for i in curr_state]
        
        # place the curr_action value in curr_action position in next state
        next_state[curr_action[0]] = curr_action[1]
        
        return next_state


    def step(self, curr_state, curr_action):
        """Takes current state and action and returns the next state, reward and whether the state is terminal. Hint: First, check the board position after
        agent's move, whether the game is won/loss/tied. Then incorporate environment's move and again check the board status.
        Example: Input state- [1, 2, 3, 4, nan, nan, nan, nan, nan], action- [7, 9] or [position, value]
        Output = ([1, 2, 3, 4, nan, nan, nan, 9, nan], -1, False)"""
        
        # call state_transition function to get the next state updated with action
        next_state = self.state_transition(curr_state, curr_action)
        
        # check whether the game has reached terminal state after AGENT's action
        is_terminal_state_reached, message = self.is_terminal(next_state)
        
        # Check if the game came to an end (Agent_win or tie. No action by Environment yet)
        if is_terminal_state_reached:
            # if AGENT wins the game reward is 10 or if the game is tied reward is 0.
            if message == "Win":                 # End the game with the reward for 'Win'
                reward = 10
                game_message = "Agent Won!"
            else:                                # End the game with the reward for 'Tie'
                reward = 0
                game_message = "It's a tie!"
                
            return (next_state, reward, is_terminal_state_reached, game_message)
        
        # Since AGENT didn't win the game or is a tie, Environment places its action
        else:
            # call action_space function to generate all possible actions for environment
            _, env_actions = self.action_space(next_state)
            
            # choose a random action from the actions available for environment
            env_action = random.choice([action for counter, action in enumerate(env_actions)])
            
            # call state_transition function to get the next state updated with action
            next_state_after_env_action = self.state_transition(next_state, env_action)
            
            # check whether the game has reached terminal state after AGENT's action
            is_terminal_state_reached, message = self.is_terminal(next_state_after_env_action)
            
            # Check if the game came to an end (Environment_win or tie)
            if is_terminal_state_reached:
                # if Environment wins the game reward is 10 or if the game is tied reward is 0.
                if message == "Win":
                    reward = -10
                    game_message = "Environment Won!"
                else:
                    reward = 0
                    game_message = "It's a tie!"
                    
            # since both Agent and environment didn't win the game, resume the game and make the reward as -1 for this step       
            else:
                reward = -1
                game_message = "Resume"
         
            return (next_state_after_env_action, reward, is_terminal_state_reached, game_message)


    def reset(self):
        return self.state
