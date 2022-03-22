#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 26 12:46:23 2021

@author: celiaberon
"""

import numpy as np
import pandas as pd


def list_to_str(seq):
    
    '''take list of ints/floats and convert to string'''
    
    seq = [str(el) for el in seq] # convert element of sequence to string
    
    return ''.join(seq) # flatten list to single string

def encode_as_ab(row, symm):
    
    '''
    converts choice/outcome history to character code where where letter represents choice and case outcome
    INPUTS:
        - row: row from pandas DataFrame containing named variables 'decision_seq' and 'reward_seq' (previous N decisions/rewards) 
        - symm (boolean): if True, symmetrical encoding with A/B for direction (A=first choice in sequence)
                          if False, R/L encoding right/left choice
    OUTPUTS:
        - (string): string of len(decision_seq) trials encoding each choice/outcome combination per trial
    
    '''
    
    if int(row.decision_seq[0]) & symm: # symmetrical mapping based on first choice in sequence 1 --> A
        mapping = {('0','0'): 'b', ('0','1'): 'B', ('1','0'): 'a', ('1','1'): 'A'} 
    elif (int(row.decision_seq[0])==0) & symm: # symmetrical mapping for first choice 0 --> A    
        mapping = {('0','0'): 'a', ('0','1'): 'A', ('1','0'): 'b', ('1','1'): 'B'} 
    else: # raw right/left mapping (not symmetrical)
        mapping = {('0','0'): 'r', ('0','1'): 'R', ('1','0'): 'l', ('1','1'): 'L'} 

    return ''.join([mapping[(c,r)] for c,r in zip(row.decision_seq, row.reward_seq)])


def add_history_cols(df, N):
    
    '''
    INPUTS:
        - df (pandas DataFrame): behavior dataset
        - N (int): number trials prior to to previous trial to sequence (history_length)
        
    OUTPUTS:
        - df (pandas DataFrame): add columns:
            - 'decision_seq': each row contains string of previous decisions t-N, t-N+1,..., t-1
            - 'reward_seq': as in decision_seq, for reward history
            - 'history': encoded choice/outcome combination (symmetrical)
            - 'RL_history': encoded choice/outcome combination (raw right/left directionality)
       
    '''
    from numpy.lib.stride_tricks import sliding_window_view
    
    df['decision_seq']=np.nan # initialize column for decision history (current trial excluded)
    df['reward_seq']=np.nan # initialize column for laser stim history (current trial excluded)

    df = df.reset_index(drop=True) # need unique row indices (likely no change)

    for session in df.Session.unique(): # go by session to keep boundaries clean

        d = df.loc[df.Session == session] # temporary subset of dataset for session
        df.loc[d.index.values[N:], 'decision_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Decision.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'reward_seq'] = \
                                    list(map(list_to_str, sliding_window_view(d.Reward.astype('int'), N)))[:-1]

        df.loc[d.index.values[N:], 'history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([True]), axis=1)

        df.loc[d.index.values[N:], 'RL_history'] = \
                                    df.loc[d.index.values[N:]].apply(encode_as_ab, args=([False]), axis=1)
        
    return df
        
        
def calc_conditional_probs(df, symm, action=['Switch'], run=0):

    '''
    calculate probabilities of behavior conditional on unique history combinations
    
    Inputs:
        df (pandas DataFrame): behavior dataset
        symm (boolean): use symmetrical history (True) or raw right/left history (False)
        action (string): behavior for which to compute conditional probabilities (should be column name in df)
        
    OUTPUTS:
        conditional_probs (pandas DataFrame): P(action | history) and binomial error, each row for given history sequence
    '''

    group = 'history' if symm else 'RL_history' # define columns for groupby function

    max_runs = len(action) - 1 # run recursively to build df that contains summary for all actions listed

    conditional_probs = df.groupby(group).agg(
        paction=pd.NamedAgg(action[run], np.mean),
        n = pd.NamedAgg(action[run], len),
    ).reset_index()
    conditional_probs[f'p{action[run].lower()}_err'] = np.sqrt((conditional_probs.paction * (1 - conditional_probs.paction))
                                                  / conditional_probs.n) # binomial error
    conditional_probs.rename(columns={'paction': f'p{action[run].lower()}'}, inplace=True) # specific column name
    
    if not symm:
        conditional_probs.rename(columns={'RL_history':'history'}, inplace=True) # consistent naming for history
    
    if max_runs == run:
    
        return conditional_probs
    
    else:
        run += 1
        return pd.merge(calc_conditional_probs(df, symm, action, run), conditional_probs.drop(columns='n'), on='history')

def sort_cprobs(conditional_probs, sorted_histories):
    
    '''
    sort conditional probs by reference order for history sequences to use for plotting/comparison
    
    INPUTS:
        - conditional_probs (pandas DataFrame): from calc_conditional_probs
        - sorted_histories (list): ordered history sequences from reference conditional_probs dataframe
    OUTPUTS:
        - (pandas DataFrame): conditional_probs sorted by reference history order
    '''
    
    from pandas.api.types import CategoricalDtype
    
    cat_history_order = CategoricalDtype(sorted_histories, ordered=True) # make reference history ordinal
    
    conditional_probs['history'] = conditional_probs['history'].astype(cat_history_order) # apply reference ordinal values to new df
    
    return conditional_probs.sort_values('history') # sort by reference ordinal values for history