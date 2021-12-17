# Import routines

import numpy as np
import math
import random
from itertools import permutations

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        
        """ Even though the cities are 1,2,3,4 and 5 the time_taken between locations in the time_matrix are starting from 0 to 4.
        Hence the action_space starts from 0 to 4.
        [(0,1), (1,0), 
         (0,2), (2,0), 
         (0,3), (3,0), 
         (0,4), (4,0), 
         (1,2), (2,1), 
         (1,3), (3,1), 
         (1,4), (4,1),
         (2,3), (3,2), 
         (2,4), (4,2), 
         (3,4), (4,3),
         (0,0)]""" 
        self.action_space = list(permutations([i for i in range(m)], 2)) + [(0,0)]
        
        # Locations       : A, B, C, D, E represented by integers 1, 2, 3, 4, 5 (start index 1). Since Time_Matrix is starting from 0, start the locations from 0.
        # Time of the day : 24 hours clock 00:00, 01:00, ..., 22:00, 23:00 represented by integers 0, 1, 2, 3, 4, ..., 22, 23
        # Day of the week : MON, TUE, WED, THU, FRI, SAT, SUN represented by integers 0, 1, 2, 3, 4, 5, 6
        self.state_space = [(a, b, c) 
                            for a in range(m) 
                            for b in range(t) 
                            for c in range(d)]
        
        # based on s = 𝑋𝑖𝑇𝑗𝐷k
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        
        if not state:
            return
        
        # initialize vector state
        state_encod = [0 for x in range (m + t + d)]
        
        # location encoding
        state_encod[state[0]] = 1
        
        # hour encoding
        state_encod[m + state[1]] = 1
        
        # week day encoding
        state_encod[m + t + state[2]] = 1
        
        return state_encod
    

    def requests(self, state):
        """Determining the number of requests basis the location.
        Use the table specified in the MDP and complete for rest of the locations"""
        
        location = state[0]
        
        requests = 0
        
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        
        if requests > 15:
            requests = 15  

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) # (0,0) is not considered as customer request
        actions = [self.action_space[i] for i in possible_actions_index]

        actions.append([0,0])
        possible_actions_index.append(self.action_space.index((0,0)))

        return possible_actions_index,actions 

    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        
        # calculate reward for the next state
        next_state, wait_time, transit_time, ride_time = self.next_state_func(state, action, Time_matrix)

        revenue_time = ride_time
        idle_time = wait_time + transit_time
        reward = (R * revenue_time) - (C * (revenue_time + idle_time))
        return reward

    # Input the current day, time and time duration and get updated day and time based on the  travel time
    def get_updated_day_time(self, time, day, duration):
        if time + int(duration) < 24:     ## no need to change day
            updated_time = time + int(duration)
            updated_day = day
        else:
            updated_time = (time + int(duration)) % 24
            days_passed = (time + int(duration)) // 24
            updated_day = (day + days_passed ) % 7
        return updated_time, updated_day


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        # Current state of driver
        current_location = state[0]
        current_time = state[1]
        current_day = state[2]  
        pickup_location = action[0]
        drop_location = action[1]
        
        if (pickup_location != 0) and (action[1] != 0) :
            pickup_location = pickup_location - 1
            drop_location = drop_location - 1
            

        # reward depends of time, lets initialize
        total_time = 0
        wait_time = 0
        ride_time = 0
        transit_time = 0

        # Three cases can be taken by driver : 
        # 1. Cab driver refuse i.e. action (0,0)
        if (pickup_location) == 0 and (drop_location == 0):
            wait_time = 1
            next_loc = current_location

        # 2. Driver wants to have pickup and is at same location
        elif pickup_location == current_location:
            ride_time = Time_matrix[current_location][drop_location][current_time][current_day]
            next_loc = drop_location

        ## 3. Driver wants to pickup and is at different location
        else:
            transit_time = Time_matrix[current_location][pickup_location][current_time][current_day]
            updated_time, updated_day = self.get_updated_day_time(current_time, current_day, transit_time)
            ride_time =  Time_matrix[pickup_location][drop_location][updated_time][updated_day]
            next_loc  = drop_location
            current_time = updated_time
            current_day = updated_day

        total_time = ride_time + wait_time
        updated_time, updated_day = self.get_updated_day_time(current_time, current_day, total_time)
        next_state = [next_loc, updated_time, updated_day]

        return next_state, wait_time, transit_time, ride_time

    def reset(self):
        self.state_init = random.choice(self.state_space)
        return self.action_space, self.state_space, self.state_init
    
c = CabDriver()