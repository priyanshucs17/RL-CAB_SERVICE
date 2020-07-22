import numpy as np
import math
import random
from itertools import permutations
from itertools import product

# Defining hyperparameters
m = 5  # number of cities, ranges from 0 ..... m-1
t = 24  # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5  # Per hour fuel and other costs
R = 9  # per hour revenue from a passenger

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space =  list(permutations([i for i in range(m)], 2)) + [(0, 0)] #list(product(range(0, m), range(0,m))) 
        self.state_space = [[x, y, z] for x in range(m) for y in range(t) for z in range(d)]
        self.state_init = random.choice(self.state_space)
        
    # define state's and action's
        self.reset()
    def get_loc(self, state):
        return state[0]

    def get_time(self, state):
        return state[1]

    def get_day(self, state):
        return state[2]

    def get_pickup(self, action):
        return action[0]

    def get_drop(self, action):
        return action[1]

    def set_loc(self, state, loc):
        state[0] = loc

    def set_time(self, state, time):
        state[1] = time

    def set_day(self, state, day):
        state[2] = day
        
    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d."""

        state_encod = np.zeros(m + t + d)
        state_encod[self.get_loc(state)] = 1
        state_encod[m + self.get_time(state)] = 1
        state_encod[m + t + self.get_day(state)] = 1
        return state_encod
    
    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
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
        # (0,0) is not consider as customer request, driver is free to refuse all customers request. Then, add index of action (0,0).
        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) + [0]
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index, actions
    
    def update_day_time(self, time, day, duration):
        
        duration = int(duration)

        if (time + duration) < 24:
            time = time + duration
            # day is unchanged
        else:
            # duration taken over consecutive days
            # convert the time to 0-23 range
            time = (time + duration) % 24 
            
            # Get the number of days
            n_days = (time + duration) // 24
            
            # Convert the day in range 0-6 
            day = (day + n_days ) % 7

        return time, day
    
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        
        # Initialize various times
        total_time   = 0     # initialize total time with 0
        transit_time = 0    # from current location to pickup location
        wait_time    = 0    # driver choose's to refuse all requests
        ride_time    = 0    # from Pick-up to drop
        next_state = []
        # creating  current location, current time, current day and requests location
        start_loc = self.get_loc(state)
        pickup_loc = self.get_pickup(action)
        drop_loc = self.get_drop(action)
        curr_hour = self.get_time(state)
        curr_day = self.get_day(state)
        """
         3 implementation's: 
           a) Refuse all requests.
           b) Driver is already at pick up point
           c) Driver is not at the pickup point.
        """    
        if ((pickup_loc== 0) and (drop_loc == 0)):
            # Refuse all requests, so wait time is 1 unit, next location is current location
            wait_time = 1
            next_loc = start_loc
        elif (start_loc == pickup_loc):
            #driver is already at pickup point, wait and transit are both are 0
            ride_time = Time_matrix[start_loc][drop_loc][curr_hour][curr_day]
            
            # next location is the drop location
            next_loc = drop_loc
        else:
            # Driver is not at the pickup point, he needs to travel pickup point first         
            transit_time = Time_matrix[start_loc][pickup_loc][curr_hour][curr_day]
            # time taken to reach pickup point
            new_hour, new_day = self.update_day_time(curr_hour, curr_day, transit_time)
            
            # driver is now at the pickup point
            # Time taken to drop the customer
            ride_time = Time_matrix[pickup_loc][drop_loc][new_hour][new_day]
            next_loc  = drop_loc

        # Calculate total time as sum of all durations
        total_time = (wait_time + transit_time + ride_time)
        #update new hour and new day
        new_hour, new_day = self.update_day_time(curr_hour, curr_day, total_time)
        
        # next_state using the next_loc and the updated new time and day.
        next_state = [next_loc, new_hour, new_day]
        
        return next_state, wait_time, transit_time, ride_time
    
    def reward_func(self, wait_time, transit_time, ride_time):
        """Takes in state, action and Time-matrix and returns the reward"""
        # transit and wait time produces no revenue, battery costs only, so they are initial times.
        travel_time = ride_time
        initial_time = wait_time + transit_time
        
        reward = (R * travel_time) - (C * (travel_time + initial_time))

        return reward
    
    def step(self, state, action, Time_matrix):
        """
        Take a trip to get rewards next step and total time spent
        """
        # next state and the various time durations
        next_state, wait_time, transit_time, ride_time = self.next_state_func(state, action, Time_matrix)

        # reward based on the different time durations
        rewards = self.reward_func(wait_time, transit_time, ride_time)
        # overall time
        total_time = wait_time + transit_time + ride_time
        
        return rewards, next_state, total_time
    
    def reset(self):
        """Return the current state and action space"""
        return self.action_space, self.state_space, self.state_init
    
    
    