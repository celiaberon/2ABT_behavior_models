#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 13:20:48 2021

@author: celiaberon
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
  
    
def get_block_position_summaries(data):
    
    bpos = pd.DataFrame()

    for group_col in ['block_pos_rev', 'blockTrial']:

        block_start, block_end = [0,20] if group_col=='blockTrial' else [-20,-1]

        summary_stats = data.groupby(group_col).agg(
            phigh = pd.NamedAgg(column = 'highPort', aggfunc = 'mean'),
            phigh_std = pd.NamedAgg(column = 'highPort', aggfunc = 'std'),
            pswitch = pd.NamedAgg(column = 'Switch', aggfunc = 'mean'),
            pswitch_std = pd.NamedAgg(column = 'Switch', aggfunc = 'std'),
            n = pd.NamedAgg(column = 'Switch', aggfunc = 'count')).loc[block_start:block_end]

        summary_stats.index.name='block_pos'

        bpos=pd.concat((bpos,summary_stats))

        
    return bpos.reset_index()
           
    
def plot_by_block_position(bpos, subset='condition', **kwargs):
    
    sns.set(style='ticks', font_scale=1.6, rc={'axes.labelsize':18, 'axes.titlesize':18}) 

    color_dict=kwargs.get('color_dict', {key:val for key, val in zip(bpos[subset].unique(), np.arange(len(bpos[subset].unique())))})
    
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10.5,3.5))
    ax1.vlines(x=0,ymin=0,ymax=1.05,linestyle='dotted',color='black')
    ax2.vlines(x=0,ymin=0,ymax=1 ,linestyle='dotted', color='black')

    for subset_iter in bpos[subset].unique(): 

        if type(color_dict[subset_iter])==np.int64:
            
            trace_color=sns.color_palette()[color_dict[subset_iter]]
            
            if subset_iter=='mouse':
                trace_color='gray'
                
        else:
            trace_color=color_dict[subset_iter]
        d = bpos.loc[bpos[subset] == subset_iter]
        
        ax1.plot(d.block_pos,d.phigh,label=subset_iter, alpha=0.8, linewidth=2, color=trace_color)
        ax1.fill_between(d.block_pos, y1=d.phigh - d.phigh_std / np.sqrt(d.n), 
                                y2=d.phigh + d.phigh_std / np.sqrt(d.n), alpha=0.2, color=trace_color)

        ax1.set_yticks([0,0.5, 1.0])

        ax2.plot(d.block_pos,d.pswitch, label=subset_iter, alpha=0.8, linewidth=2, color=trace_color)
        ax2.fill_between(d.block_pos,y1=d.pswitch - d.pswitch_std / np.sqrt(d.n), 
                                    y2=d.pswitch + d.pswitch_std / np.sqrt(d.n), alpha=0.2, color=trace_color)
        
        ax2.set_yticks(np.arange(0,0.6,step=0.1))#[0,0.1, 0.4])
    ax1.set(xlim=(-10,20), ylim=(0,1), xlabel='Block Position', ylabel='P(high port)')
    ax2.set(xlim=(-10,20), ylim=(0,np.max(bpos.pswitch)+0.05), xlabel='Block Position', ylabel='P(switch)') 
    
    if len(bpos[subset].unique())<5:
        ax1.legend(loc=[0.5,-0.03], fontsize=16,frameon=False)
    sns.despine()
    plt.tight_layout()
    
    
def plot_scatter(df_mouse, df_model):
    
    sns.set(style='ticks', font_scale=1.6, rc={'axes.labelsize':18, 'axes.titlesize':18})   
    sns.set_palette('dark')
    
    plt.figure(figsize=(4,4))
    plt.subplot(111, aspect='equal')
    plt.scatter(df_mouse.pswitch, df_model.pswitch, alpha=0.6, edgecolor=None, linewidth=0)
    plt.plot([0, 1], [0, 1], ':k')
    
    plt.xlabel('P(switch)$_{mouse}$')
    plt.ylabel('P(switch)')
    plt.xticks(np.arange(0, 1.1, 0.5))
    plt.yticks(np.arange(0, 1.1, 0.5))
    
    plt.tight_layout()
    sns.despine()
  
    
def plot_sequences(df, overlay=[], **kwargs):
    
    sns.set(style='ticks', font_scale=1.7, rc={'axes.labelsize':20, 'axes.titlesize':20})
    sns.set_palette('deep')

    overlay_label = kwargs.get('overlay_label', '')
    main_label = kwargs.get('main_label', '')
    yval = kwargs.get('yval','pswitch')
    
    df = df.astype('object') # to deal with histories being treated as categorical from sorting
    
    fig, ax = plt.subplots(figsize=(14,4.2))
    if len(overlay)>0:
        overlay = overlay.astype('object')
        sns.barplot(x='history',y=yval, data=overlay, label=overlay_label, color=sns.color_palette()[0], ax=ax, alpha=1.0)
        ax.errorbar(x='history',y=yval, yerr=yval+'_err', data=overlay, fmt=' ', label=None, color=sns.color_palette('dark')[0])
        
    sns.barplot(x='history',y=yval,data=df, color='k', alpha=kwargs.get('alpha',0.4), label=main_label, ax=ax, edgecolor='gray')
    ax.errorbar(x='history',y=yval, yerr=yval+'_err', data=df, fmt=' ', color='k', label=None)

    
    if len(overlay_label)>0:
        ax.legend(loc='upper left', frameon=False)
    ax.set(xlim=(-1,len(df)), ylim=(0,1), ylabel='P(switch)', title=kwargs.get('title', None))
    plt.xticks(rotation=90)
    sns.despine()
    plt.tight_layout()
    
    
def internal_prob(a, b, n):
    
    return np.nansum(a * b * n) / np.nansum(n)


def calc_confusion_matrix(df_mouse, col, df_model=None):

    #[[actual repeat * predict repeat, actual repeat * predict switch],
    #[actual switch * predict repeat, actual switch * predict swich]]
    # and can sub in right / left
    
    if df_model is None:
        df_model = df_mouse.copy()
    else:
        assert(np.all(df_mouse.history.values == df_model.history.values))
    
    N = df_mouse.n.values # same counts for model
    a = df_mouse[col].values
    b = df_model[col].values

    raw_confusion = np.array([[internal_prob(1-a, 1-b, N), internal_prob(1-a, b, N)],
                               [internal_prob(a, 1-b, N), internal_prob(a, b, N)]])

    norm_confusion = raw_confusion / raw_confusion.sum(axis=1)[:,np.newaxis]

    return norm_confusion


def plot_confusion(df, df_model, cm_fig=None, col='pswitch', color='Blues', seq_nback=3, delta=True):
    
    sns.set(style='white', font_scale=1.3, rc={'axes.labelsize':16, 'axes.titlesize':16})
    
    if cm_fig is None:
        cm_fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.4, 2.2))
    else:
        ax = cm_fig.add_subplot(1, 2+delta, 2+delta)
        cm_fig.set_size_inches(8.4, 2.5)

    cm = calc_confusion_matrix(df, col, df_model)
        
    ax.imshow(cm, cmap=color)

    fmt='.2f'
    thresh = cm.max()/ 2.
    for i, row in enumerate(cm):
            
            for j, square in enumerate(row):
                
                ax.text(j, i, format(square, fmt),
                        ha="center", va="center",
                        color="white" if square > thresh else "black")
                
    column_dict = {'pswitch': ['repeat', 'switch'], 'pdecision':['left','right']}
    
    ax.set_xticks((0,1))
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('top') 
    ax.set_xticklabels(('{} '.format(column_dict[col][0]),' {}'.format(column_dict[col][1])))
    ax.set_yticks((0,1))
    ax.set_yticklabels(('{}'.format(column_dict[col][0]),'{}'.format(column_dict[col][1])))
    ax.set(xlabel='predicted', ylabel='actual', ylim=(-0.5, 1.5)) 
    ax.invert_yaxis()
    plt.tick_params(top=False, pad=-2)
    plt.tight_layout()
    
    return cm_fig