#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  1 14:36:04 2021

@author: celiaberon
"""

import numpy as np
import pandas as pd
import itertools

import plot_models_v_mouse as bp
import conditional_probs as cprobs
import model_policies as models
from sklearn.utils import resample


def pull_sample_dataset(session_id_list, data):
        
    '''
    INPUTS:
        - session_id_list (list): list of session names
        - data (pandas DataFrame): dataset
            
    OUTPUTS: 
        - sample_features (list of lists): [choices, rewards] x session
        - sample_target (list of lists): [target port] x session
        - sample_block_pos_core (pandas DataFrame): in the same format as DATA, containing only sessions in/sorted by session_id_list 
            
    Note: can be used to sample for train and test sets or for resampling from dataset for bootstrapping
    '''
    
    # choices and rewards (second idx, {0,1}) by session (first idx {0:nSessions}) for models
    sample_features = [[data[data.Session==session].Decision.values.astype('int'), \
                        data[data.Session==session].Reward.values.astype('int')] for session in session_id_list]
    sample_target = [data[data.Session==session].Target.values.astype('int') for session in session_id_list] # for expected reward only

    # makde test_df ordered same as test_sessions
    sample_block_pos_core = pd.concat([data[data.Session == session] for session in session_id_list] ).reset_index(drop=True)
    
    return sample_features, sample_target, sample_block_pos_core



def reconstruct_block_pos(blocks, model_choice, model_switch):
    
    '''
    takes mouse dataframe and replaces Switch, Decision, and highPort columns 
    with model predictions and get summary at each block position
    
    INPUTS:
        - blocks (pandas DataFrame): df with row for each trial, includes block position column
        - model_choice (nested lists): list of model choice predictions for each session
        - model_switch (nested lists): as with model_choice, for switch predictions
        
    OUTPUTS:
        - block_pos_model (pandas DF): same as BLOCKS, but with Switch, Decision, and highPort columns
                                       replaced with model predictions; model label marks model predictions
    '''
    
    block_pos_model = blocks.copy() 
    block_pos_model['model'] = 'model' # label all rows to fill as model predictions

    block_pos_model['Switch']=list(itertools.chain(*model_switch)) # fill with model switch predictions
    block_pos_model['Decision'] = list(itertools.chain(*model_choice)) # fill with model choice predictions
    block_pos_model['highPort']= list(itertools.chain(*model_choice)) == block_pos_model.Target # fill with model higher prob port
    
    return block_pos_model # return model version of data


def build_model_dfs(block_pos_model):
    
    '''initializes df for each form of analysis'''
            
    block_pos_model_summary = bp.get_block_position_summaries(block_pos_model)
    
    symm_cprobs_model = cprobs.calc_conditional_probs(block_pos_model, symm=True, action=['Switch', 'Decision']).reset_index()    

    port_cprobs_model = cprobs.calc_conditional_probs(block_pos_model, symm=False, action=['Switch','Decision']).reset_index()

    return symm_cprobs_model, port_cprobs_model, block_pos_model_summary


def append_model_reps(block_pos_model, df_reps=None):
    
    '''builds up dataframes across repetitions of model runs'''

    cprobs_symm, cprobs_port, bpos_model = build_model_dfs(block_pos_model)
    
    phigh_reps, pswitch_reps, cprobs_symm_reps, cprobs_port_reps = df_reps #unpack df_reps
    
    phigh_reps = phigh_reps.merge(bpos_model[['block_pos','phigh']], on='block_pos', how='left', sort=False, suffixes=('', str(len(phigh_reps.columns))))
    pswitch_reps = pswitch_reps.merge(bpos_model[['block_pos', 'pswitch']], on='block_pos', how='left', sort=False, suffixes=('', str(len(pswitch_reps.columns))))
    cprobs_symm_reps = cprobs_symm_reps.merge(cprobs_symm[['history', 'pswitch']], on='history', how='left', sort=False, suffixes=('', str(len(cprobs_symm_reps.columns))))
    cprobs_port_reps = cprobs_port_reps.merge(cprobs_port[['history', 'pdecision']], on='history', how='left', sort=False, suffixes=('', str(len(cprobs_port_reps.columns))))
    
    return phigh_reps, pswitch_reps, cprobs_symm_reps, cprobs_port_reps

    
def reps_wrapper(model_func, session_list, data, n_reps, action_policy='stochastic', bs=True, **kwargs):
    
    '''
    Resamples sessions to run repetitions of model predictions using given model
    
    INPUTS:
        - model_func (function): partial model function with fit parameters given
        - session_list (list): list of sessions to be resampled
        - data (pandas DF): mouse behavior data 
        - n_reps (int): number of repetitions to run
        - action_policy (string): 'greedy', 'stochastic', 'softmax'
        - bs (bool): True if resampling, False if reps on same sessions
        **kwargs:
            - inv_temp (float): temperature parameter for softmax policy
            
    OUTPUTS:
        - bpos_model (pandas DF): summary across reps of P(high port) and P(switch) at each block position
        - cprobs_symm (pandas DF): summary across reps of P(switch | history) for symmetrical history 
        - cprobs_port (pandas DF): as above, but for lateralized history (right/left directionality preserved)
    '''
    
    phigh = bp.get_block_position_summaries(data)[['block_pos']].copy() # block_pos column only
    pswitch = phigh.copy() # block_pos column only
    
    cprobs_symm = cprobs.calc_conditional_probs(data, symm=True, action=['Switch', 'Decision'])
    cprobs_symm = cprobs_symm.sort_values(by='pswitch')[['history']].copy() # full dataset sorted histories 
    
    cprobs_port = cprobs.calc_conditional_probs(data, symm=False, action=['Switch','Decision'])
    cprobs_port = cprobs_port.sort_values(by='pswitch')[['history']].copy() # full dataset sorted histories
    
    for i in range(n_reps):
        
        if i%100 == 0: print(f'rep {i}')

        if bs:
            resampled_sessions = resample(session_list) # resample test dataset with replacement
        else:
            resampled_sessions = session_list # if just running reps on test dataset without bootstrapping
        resampled_choice_reward, _, resampled_block_pos_core = pull_sample_dataset(resampled_sessions, data)
        model_probs = model_func(resampled_choice_reward) # calculate model probs on resampled dataset
        model_choices, model_switches = models.model_to_policy(model_probs, resampled_choice_reward, policy=action_policy, **kwargs)
        block_pos_model = reconstruct_block_pos(resampled_block_pos_core, model_choices, model_switches)
        
        # append to existing dfs from current resample
        phigh, pswitch, cprobs_symm, cprobs_port = append_model_reps(block_pos_model, df_reps=[phigh, pswitch, cprobs_symm, cprobs_port])

    block_pos = phigh.pop('block_pos') # new df with block positions
    pswitch = pswitch.drop(columns='block_pos')
    bpos_model = pd.DataFrame(data={'block_pos':block_pos, 'phigh': phigh.mean(axis=1),'phigh_std':phigh.std(axis=1),#/np.sqrt(n_reps), 
                                    'pswitch':pswitch.mean(axis=1), 'pswitch_std':pswitch.std(axis=1), 'n':n_reps}) #/np.sqrt(n_reps)}


    history = cprobs_symm.pop('history')
    cprobs_symm = pd.DataFrame(data={'history':history, 'n_reps':n_reps,
                                     'pswitch': np.nanmean(cprobs_symm,axis=1),'pswitch_err':np.nanstd(cprobs_symm,axis=1)})#/np.sqrt(n_reps)})
    
    history = cprobs_port.pop('history')
    cprobs_port = pd.DataFrame(data={'history':history, 'n_reps':n_reps,
                                     'pdecision': np.nanmean(cprobs_port,axis=1),'pdecision_err':np.nanstd(cprobs_port,axis=1)})#/np.sqrt(n_reps)})
    
    
    return bpos_model, cprobs_symm, cprobs_port