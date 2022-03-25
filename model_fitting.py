#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 16:21:06 2021

@author: celiaberon
"""


import jax.numpy as jnp
from jax import jit, lax, value_and_grad
from tqdm.auto import trange
import numpy as np


def fit_with_sgd(ll_func, training_data, num_steps=10000, step_size=1e-1, init_parameters = (1.0, 1.0, 1.0)):
    
    '''
    fit behavior model with sgd, basic architecture
    
    INPUTS:
        - ll_func (function): log likelihood function for given model
        - traning_data (nested lists): [choices, rewards] by session 
        - num_steps (int)
        - step_size (float)
        - init_parameters (tuple): starting parameters, varies in length by model
        
    OUTPUTS:
        - (np array) optimized parameters
        - nll: negative log likelihood
    '''
    
    # simple gradient ascent algorithm
    from jax.example_libraries.optimizers import sgd

    init_fun, update_fun, get_params = sgd(step_size)
    opt_state = init_fun(init_parameters)

    loss_fn = lambda parameters: -ll_func(parameters, training_data)
    loss_fn_and_grad = jit(value_and_grad(loss_fn))

    def step(itr, opt_state):
        value, grads = loss_fn_and_grad(get_params(opt_state))
        opt_state = update_fun(itr, grads, opt_state)
        return value, opt_state

    for i in trange(num_steps, disable=True):
        nll, opt_state = step(step, opt_state)
        if i % int(round(num_steps/4)) == 0:
            print("iteration ", i, "neg ll: ", nll)
            
    return np.asarray(opt_state[0]), nll   


'''RECURSIVELY FORMULATED LOGISTIC REGRESSION'''

@jit

def _log_prob_single_rflr(parameters, choices, rewards):
    
    alpha, beta, tau = parameters  # unpack parameters
    gamma = jnp.exp(-1 / tau)
    
    def update(carry, x):
        # unpack the carry
        ll, phi = carry

        # unpack the input
        prev_choice, choice, reward = x

        # update
        psi = phi + alpha * (2 * prev_choice - 1)
        ll += choice * psi - jnp.log(1 + jnp.exp(psi))
        phi = gamma * phi + beta * reward * (2 * choice - 1)

        new_carry = (ll, phi) 
        return new_carry, None
    
    
    ll = 0.0
    phi = beta * rewards[0] * (2 * choices[0] - 1)
    (ll, phi), _ = lax.scan(update, (ll, phi), (choices[:-1], choices[1:], rewards[1:]))
    
    return ll
    
def log_probability_rflr(parameters, sessions):
    
    # compute probability of next choice
    ll = 0.0
    n = 0
    for choices, rewards in sessions:        
        # initialize "belief state" for this session
        
        ll += _log_prob_single_rflr(parameters, choices, rewards)
        n += len(choices) - 1
            
    return ll / n
