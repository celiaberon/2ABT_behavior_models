#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  4 11:29:49 2021

@author: celia
"""

import numpy as np
from scipy.special import expit as sigmoid
import numpy.random as npr
from scipy.special import logit

def observe_reward(choice, state, phigh):
    
    if choice==state:
        return np.random.choice(2,p=[1-phigh, phigh])
    else:
        return np.random.choice(2, p=[phigh,1-phigh])

def markov_process(curr_state,tprob):
    
    if curr_state:
        return np.random.choice(2,p=[tprob, 1-tprob])
    else:
        return np.random.choice(2,p=[1-tprob, tprob])
            
    
def make_choice(psi):
    
    return int(np.random.rand() < sigmoid(psi))
    
def rflr_simulation(rflr_params, task_params, nTrials=30000):
    
    alpha, beta, tau = rflr_params  # unpack parameters
    phigh, tprob = task_params
    
    gamma = np.exp(-1 / tau)
    
    nSessions = round(nTrials/750) # setting session length by mean mouse session
            
    nRewards=0 # cumulative reward count   
    N=0 # keep track of how many trials
    
    sessions = []
    session_states = []
    
    for iSession in range(1,nSessions):
        
        nTrials = np.random.randint(650,850)
    
        choices = [np.random.randint(2)] # randomly initialize first choice
        states = [np.random.randint(2)] # randomly initialize first state
        rewards = [observe_reward(choices[0], states[0], phigh)]
        psis = []
        # initialize "belief state"
        phi = beta * rewards[0] * (2 * choices[0] - 1)
        
        nRewards += rewards[0]
        N+=1
        
        for t in range(1, nTrials):

            states.append(markov_process(states[t-1], tprob))
            
            # update belief state for next time step
            psi = phi + (alpha * (2*choices[t-1]-1)) # compute probability of next choice

            # make choice and observe outcome off belief state
            choices.append(make_choice(psi))
            rewards.append(observe_reward(choices[t], states[t], phigh)) # now have moved to current time step
            psis.append(psi)
            
            phi = gamma * phi + (beta*(rewards[t] * (2*choices[t]-1))) # update evidence

            nRewards+=rewards[t]
            N+=1
            
        sessions.append([np.array(choices), np.array(rewards), np.array(psis)])
        session_states.append(states)
        
    return nRewards/N, sessions, session_states


class Mouse(object):
    """
    A mouse is just an agent that chooses left or right based on past experience and possibly some model 
    of the world.  It maintains some state that summarizes the past experience and updates that state
    when it receives new feedback; i.e. the outcomes of its choices.
    """
    def make_choice(self):
        raise NotImplementedError
        
    def receive_feedback(self, choice, reward):
        raise NotImplementedError

class BayesianMouse(Mouse):
    """
    This mouse maintains an estimate of the posterior distribution of the 
    rewarded port based on past choices and rewards.
    """
    def __init__(self, params):
        """
        Specify the HMM model parameters including 'p_switch' and 'p_reward'
        """
        # transition matrix specifies prob for each (current state, next state) pair
        p_switch = params['p_switch']
        self.transition_matrix = (1 - p_switch) * np.eye(2)
        self.transition_matrix += p_switch * (1 - np.eye(2))
        
        # reward probability specifies prob for each (choice, state) pair
        p_reward = params['p_reward']
        self.reward_probability = p_reward * np.eye(2)
        self.reward_probability += (1 - p_reward) * (1 - np.eye(2))
        self.posterior = 0.5 * np.ones(2)

    def make_choice(self, policy):
        
        prediction = np.dot(self.transition_matrix.T, self.posterior)
        
        if policy=='greedy':
            return np.where(prediction==prediction.max())[0][0]
        elif policy=='thompson':
            return npr.rand() < prediction[1]

        
    def receive_feedback(self, choice, reward):
        # For simulation only: update posterior distribution given new information
        assert self.posterior.shape == (2, )
        pr = self.reward_probability[int(choice)]
        lkhd = pr if reward else (1 - pr)
        
        self.posterior = np.dot(self.transition_matrix.T, self.posterior) * lkhd
        self.posterior /= self.posterior.sum()
        assert self.posterior.shape == (2, )
        
        
def simulate_experiment(params, mouse, states, policy, sticky=False):
    """
    Simulate an experiment with a given set of parameters and a mouse model.
    
    'params' is a dictionary with keys:
        'p_switch':  the probability that the rewarded port switches
        'p_reward':  the probability that reward is delivered upon correct choice
        
    'mouse' is an instance of a Mouse object defined below
    
    """
    sessions = []

    # Run the simulation
    for session_states in states: # switched from trange for reps
        
        choices, rewards, beliefs = [],[], []
        mouse.posterior = 0.5 * np.ones(2) # reset posterior for each session
        for state in session_states:
            # Make choice according to policy
            choices.append(mouse.make_choice(policy))

            # Deliver stochastic reward
            if choices[-1] == state:
                rewards.append(npr.rand() < params["p_reward"])
            else:
                rewards.append(npr.rand() < (1 - params["p_reward"]))
            
            mouse.receive_feedback(choices[-1], rewards[-1]) # update posterior
            
            if (sticky==True) & (len(choices)>2):
                mouse.update_stickiness(choices[-2:])
            beliefs.append(mouse.posterior)
        sessions.append([np.array(choices, dtype='int'), np.array(rewards, dtype='int'), np.array(beliefs)])
    
    return sessions