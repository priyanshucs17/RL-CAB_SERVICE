# RL-CAB_SERVICE
## Problem Statement:
You are hired as a Sr. Machine Learning Er. at SuperCabs, a leading app-based cab provider in a large Indian metro city. In this highly competitive industry, __retention of good cab drivers__ is a crucial business driver, and you believe that a sound __RL-based system for assisting cab drivers__ can potentially retain and attract new cab drivers. 

Cab drivers, like most people, are incentivised by a healthy growth in income. The goal of your project is to build an RL-based algorithm which can __help cab drivers maximise their profits__ by improving their decision-making process on the field.

## The Need for Choosing the 'Right' Requests:
Most drivers get a healthy number of ride requests from customers throughout the day. But with the recent hikes in electricity prices (all cabs are electric), many drivers complain that although their revenues are gradually increasing, their profits are almost flat. Thus, it is important that drivers choose the 'right' rides, i.e. choose the rides which are likely to __maximise the total profit__ earned by the driver that day. 

For example, say a driver gets three ride requests at 5 PM. The first one is a long-distance ride guaranteeing high fare, but it will take him to a location which is unlikely to get him another ride for the next few hours. The second one ends in a better location, but it requires him to take a slight detour to pick the customer up, adding to fuel costs. Perhaps the best choice is to choose the third one, which although is medium-distance, it will likely get him another ride subsequently and avoid most of the traffic. 

There are some basic rules governing the ride-allocation system. If the cab is already in use, then the driver won’t get any requests. Otherwise, he may get multiple request(s). He can either decide to take any one of these requests or can go ‘offline’, i.e., not accept any request at all.

## Markov Decision Process:
Taking long-term profit as the goal, you propose a method based on __reinforcement learning__ to optimize taxi driving strategies for profit maximization. This optimization problem is formulated as a __Markov Decision Process__.

In this project, you need to create the environment and an RL agent that learns to choose the best request. You need to train your agent using vanilla __Deep Q-learning__ (DQN) only and __NOT a double DQN__. You have learnt about the two architectures of DQN (shown below) - you are free to choose any of these.

There’s a renowned paper called ‘Deep Reinforcement Learning for List-wise Recommendations’ by __Xiangyu Zhao__, __Liang Zhang__, __Zhuoye Ding__. They have mentioned a few recommendations on how to select the Q-network architecture. You can download the paper from below (refer to the highlighted section 1.2 - Architecture Selection). But referring to the paper is an __optional part__ of this project.



 


PGDML_CASE-STUDY
