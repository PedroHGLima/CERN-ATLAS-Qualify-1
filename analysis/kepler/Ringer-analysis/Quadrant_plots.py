#!/usr/bin/env python
# coding: utf-8

# # Kepler Framework Examples

# In[1]:


from kepler.pandas.menu       import ElectronSequence as Chain
from kepler.pandas.readers    import load, load_in_loop
from kepler.pandas.decorators import create_ringer_v8_decorators, create_ringer_v9_decorators, RingerDecorator
from kepler.pandas.decorators import create_ringer_v8_new_decorators, create_ringer_v8_half_fast_decorators, create_ringer_v8_34_decorators, create_ringer_v8_half_decorators, create_ringer_v8_14_decorators

import kepler
from tqdm import tqdm
import rootplotlib as rpl
import mplhep as hep
import root_numpy
import ROOT
ROOT.gStyle.SetOptStat(0);
import array

import numpy as np
import pandas as pd
import collections
import os
from pprint import pprint
from copy import deepcopy
import gc


import matplotlib.pyplot as plt
from matplotlib import gridspec
get_ipython().run_line_magic('matplotlib', 'inline')

import mplhep as hep

import warnings
warnings.filterwarnings('ignore')
plt.style.use(hep.style.ROOT)
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# # Running for Zee samples

# In[9]:


## codigo original
# dpath = '/home/jodafons/public/cern_data/new_files/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins'
# dpath+= '/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins_et{ET}_eta{ETA}.npz'
# paths = []
# for et in range(5):
#     for eta in range(5):
#         paths.append( dpath.format(ET=et,ETA=eta) )

real_run = False


# ## Load Data

# In[10]:


# codigo modificado
dpath = '/home/pedro.lima/data/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins'
dpath+= '/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins_et{ET}_eta{ETA}.npz'
#                                               \
#                                           very loose
paths = []

dev = False
#    False carrega todas as regioes

if dev:
    range_et = [4]   #leblon = 4
    range_eta = [0]  #leblon = 0
else:
    range_et = range(5)
    range_eta = range(5)

for et in range_et:
    for eta in range_eta:
        paths.append( dpath.format(ET=et,ETA=eta) )
# look here: https://github.com/ringer-softwares/kolmov/blob/master/kolmov/utils/constants.py


# In[11]:


drop_columns = [
                    'RunNumber', 
                    #'avgmu',
                    #'trig_L2_cl_et',
                    #'trig_L2_cl_eta',
                    #'trig_L2_cl_phi',
                    #'trig_L2_cl_reta',
                    #'trig_L2_cl_ehad1', 
                    #'trig_L2_cl_eratio',
                    #'trig_L2_cl_f1', 
                    #'trig_L2_cl_f3', 
                    #'trig_L2_cl_weta2', 
                    #'trig_L2_cl_wstot', 
                    'trig_L2_cl_e2tsts1',
                    'trig_L2_el_hastrack',
                    #'trig_L2_el_pt',
                    'trig_L2_el_eta',
                    'trig_L2_el_phi',
                    'trig_L2_el_caloEta',
                    'trig_L2_el_trkClusDeta',
                    'trig_L2_el_trkClusDphi',
                    'trig_L2_el_etOverPt',
                    'trig_EF_cl_hascluster',
                    #'trig_EF_cl_et',
                    'trig_EF_cl_eta',
                    'trig_EF_cl_etaBE2',
                    'trig_EF_cl_phi',     
                    'trig_EF_el_hascand',
                    #'trig_EF_el_et',
                    'trig_EF_el_eta',
                    'trig_EF_el_etaBE2',
                    'trig_EF_el_phi',
                    'trig_EF_el_rhad1',
                    'trig_EF_el_rhad',
                    'trig_EF_el_f3',
                    'trig_EF_el_weta2',
                    'trig_EF_el_rphi',
                    'trig_EF_el_reta',
                    'trig_EF_el_wtots1',
                    'trig_EF_el_eratio',
                    'trig_EF_el_f1',
                    'trig_EF_el_hastrack',
                    'trig_EF_el_deltaEta1',
                    'trig_EF_el_deltaPhi2',
                    'trig_EF_el_deltaPhi2Rescaled',
                    #'trig_EF_el_lhtight',
                    #'trig_EF_el_lhmedium',
                    #'trig_EF_el_lhloose',
                    #'trig_EF_el_lhvloose', 
                    # Offline variables
                    #'el_et',
                    #'el_eta',
                    'el_etaBE2',
                    #'el_phi',
                    # offline shower shapers
                    #'el_rhad1',
                    #'el_rhad',
                    #'el_f3',
                    #'el_weta2',
                    #'el_rphi',
                    #'el_reta',
                    #'el_wtots1',
                    #'el_eratio',
                    #'el_f1',
                    # offline track
                    #'el_hastrack',
                    'el_numberOfBLayerHits',
                    'el_numberOfPixelHits',
                    'el_numberOfTRTHits',
                    #'el_d0',
                    #'el_d0significance',
                    #'el_eProbabilityHT',
                    'el_trans_TRT_PID',
                    #'el_deltaEta1',
                    'el_deltaPhi2',
                    #'el_deltaPhi2Rescaled',
                    #'el_deltaPOverP',
                    #'el_lhtight',
                    #'el_lhmedium',
                    #'el_lhloose',
                    #'el_lhvloose',
                    'el_TaP_Mass',
                    #'el_TaP_deltaR',
                ] 

# variaveis dos aneis...para plotar o perfil médio, preciso deixar
#drop_columns.extend( ['trig_L2_cl_ring_%d'%i for i in range(100)] )


# In[12]:


os.environ['RINGER_TUNING_PATH']='/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/tunings'

decorators = create_ringer_v8_new_decorators()
decorators += create_ringer_v8_half_fast_decorators()
#decorators+= create_ringer_v9_decorators()
#decorators += create_ringer_v8_34_decorators()
decorators += create_ringer_v8_14_decorators()


# ## Setup Chains

# ivarloose - é o isolamento loose. Olhar https://twiki.cern.ch/twiki/bin/view/Atlas/TriggerNamingRun2    https://twiki.cern.ch/twiki/bin/view/Atlas/TriggerMenuConvention
# 
# ivarloose (Run2 que deve ser mantido para o Run3)- HLT isolation: ptvarcone20/ET<0.1
# 
# O pid_name da cadeia está marcado no `lh*` , onde * é o pid_name (ponto de operação da cadeia)

# In[13]:


# create my chain
chains = [
            Chain( "HLT_e24_lhtight_nod0_noringer_ivarloose" , L1Seed = 'L1_EM22VHI'),
            Chain( "HLT_e24_lhtight_nod0_ringer_v8_new_ivarloose", L1Seed = 'L1_EM22VHI' , l2calo_column = 'ringer_v8_new_tight' ),
            Chain( "HLT_e24_lhtight_nod0_ringer_v8_half_ivarloose", L1Seed = 'L1_EM22VHI' , l2calo_column = 'ringer_v8_half_tight' ),
            #Chain( "HLT_e24_lhtight_nod0_ringer_v8_34_ivarloose", L1Seed = 'L1_EM22VHI' , l2calo_column = 'ringer_v8_34_tight' ),
            Chain( "HLT_e24_lhtight_nod0_ringer_v8_14_ivarloose", L1Seed = 'L1_EM22VHI' , l2calo_column = 'ringer_v8_14_tight' ),

            Chain( "HLT_e26_lhtight_nod0_noringer_ivarloose" , L1Seed = 'L1_EM22VHI'),
            Chain( "HLT_e26_lhtight_nod0_ringer_v8_new_ivarloose", L1Seed = 'L1_EM22VHI' , l2calo_column = 'ringer_v8_new_tight' ),
            Chain( "HLT_e26_lhtight_nod0_ringer_v8_half_ivarloose", L1Seed = 'L1_EM22VHI' , l2calo_column = 'ringer_v8_half_tight' ),
            #Chain( "HLT_e26_lhtight_nod0_ringer_v8_34_ivarloose", L1Seed = 'L1_EM22VHI' , l2calo_column = 'ringer_v8_34_tight' ),
            Chain( "HLT_e26_lhtight_nod0_ringer_v8_14_ivarloose", L1Seed = 'L1_EM22VHI' , l2calo_column = 'ringer_v8_14_tight' ),

            Chain( "HLT_e60_lhmedium_nod0_noringer_L1EM24VHI" , L1Seed = 'L1_EM24VHI'),
            Chain( "HLT_e60_lhmedium_nod0_ringer_v8_new_L1EM24VHI", L1Seed = 'L1_EM24VHI' , l2calo_column = 'ringer_v8_new_medium'),
            Chain( "HLT_e60_lhmedium_nod0_ringer_v8_half_L1EM24VHI", L1Seed = 'L1_EM24VHI' , l2calo_column = 'ringer_v8_half_medium'),
            #Chain( "HLT_e60_lhmedium_nod0_ringer_v8_34_L1EM24VHI", L1Seed = 'L1_EM24VHI' , l2calo_column = 'ringer_v8_34_medium'),
            Chain( "HLT_e60_lhmedium_nod0_ringer_v8_14_L1EM24VHI", L1Seed = 'L1_EM24VHI' , l2calo_column = 'ringer_v8_14_medium'),

          
            Chain( "HLT_e140_lhloose_nod0_noringer"  , L1Seed = 'L1_EM24VHI'),
            Chain( "HLT_e140_lhloose_nod0_ringer_v8_new" , L1Seed = 'L1_EM24VHI', l2calo_column = 'ringer_v8_new_loose'),
            Chain( "HLT_e140_lhloose_nod0_ringer_v8_half" , L1Seed = 'L1_EM24VHI', l2calo_column = 'ringer_v8_half_loose'),
            #Chain( "HLT_e140_lhloose_nod0_ringer_v8_34" , L1Seed = 'L1_EM24VHI', l2calo_column = 'ringer_v8_34_loose'),
            Chain( "HLT_e140_lhloose_nod0_ringer_v8_14" , L1Seed = 'L1_EM24VHI', l2calo_column = 'ringer_v8_14_loose'),
]


# ## Read all bins

# In[14]:


table = load_in_loop( paths, drop_columns=drop_columns, decorators=decorators, chains=chains )


# In[15]:


if True:
    print(table.columns.to_list())


# In[16]:


table.head()


# ## Quadrant plots

# ### Quads for $R_{\eta}$

# In[28]:


def quad_reta(step_num=0, electron=True):
    var = 'trig_L2_cl_reta'

    for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][step_num]]:
        plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/quad/reta'

        chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                      'e26_lhtight_nod0_{RINGER}_ivarloose',
                      'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                      'e140_lhloose_nod0_{RINGER}'
                     ]

        bins = int(np.sqrt(table.loc[table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)]].shape[0]))

        for chain in chain_list:
            chain = step + '_' + chain
            print('Processing %s ...' %(chain))

            # create quadrant tables
            #first_quad = table.loc[(table[chain.format(RINGER = alg1)]==1) & (table[chain.format(RINGER = alg2)]==1)]['el_eta']
            #second_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] ==1)]['el_eta']
            #third_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']
            #fourth_quad = table.loc[(table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']

            # calculate counts for each quad
            ## constructing the tables here helps in saving memory
            upper_limit = 1.1
            lower_limit = 0.8
            
            [count_total, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_first, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] ==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_second, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] ==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_third, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] !=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_fourth, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] !=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            plt.clf()

            # create fig
            fig = plt.figure(figsize = (20,20), constrained_layout=False)
            gs = fig.add_gridspec(nrows=3, ncols=1)
            ax0 = fig.add_subplot(gs[:-1])
            ax1 = fig.add_subplot(gs[-1])

            # plot each quad
            ax0.plot(bins[:-1], count_first, 'o', color = 'black', label='Both approve')
            ax0.plot(bins[:-1], count_second, 'o', color = 'red', label=lbl1)
            ax0.plot(bins[:-1], count_third, 'o', color = 'grey', label='Both reject')
            ax0.plot(bins[:-1], count_fourth, 'o', color = 'blue', label=lbl2)
            ax0.set(ylabel='Count [$log_{10}$]')
            ax0.set_yscale('log')
            ax0.legend(frameon=True, framealpha=True)
            ax0.grid()

            # plot disagreement
            ax1.plot(bins[1], count_second[1]/count_total[1]*100, 'o', color = (0,0,0,0))
            ax1.plot(bins[:-1], count_second/count_total*100, 'o', color = 'red', label = lbl1)
            ax1.plot(bins[:-1], count_fourth/count_total*100, 'o', color = 'blue', label = lbl2)
            ax1.legend(['%s_%s_%s_nod0' %(step, chain.split('_')[1], chain.split('_')[2])], loc='best')
            ax1.set(ylabel='Disagreement [%]', xlabel='$R_{\eta}$')

            # save fig
            part = 'electron' if electron else 'jet'
            print('Saving ' + plot_path + '/quad_reta_%s_%s_%s_%s ...'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_reta_%s_%s_%s_%s.png'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_reta_%s_%s_%s_%s.pdf'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.show()


# ### Quads for $E_{ratio}$

# In[29]:


def quad_eratio(step_num=0, electron=True):
    var = 'trig_L2_cl_eratio'

    for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][step_num]]:
        plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/quad/eratio'

        chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                      'e26_lhtight_nod0_{RINGER}_ivarloose',
                      'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                      'e140_lhloose_nod0_{RINGER}'
                     ]

        bins = int(np.sqrt(table.loc[table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)]].shape[0]))

        for chain in chain_list:
            chain = step + '_' + chain
            print('Processing %s ...' %(chain))

            # create quadrant tables
            #first_quad = table.loc[(table[chain.format(RINGER = alg1)]==1) & (table[chain.format(RINGER = alg2)]==1)]['el_eta']
            #second_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] ==1)]['el_eta']
            #third_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']
            #fourth_quad = table.loc[(table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']

            # calculate counts for each quad
            ## constructing the tables here helps in saving memory
            upper_limit = 1.04
            lower_limit = 0.5
            
            [count_total, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_first, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] ==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_second, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] ==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_third, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] !=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_fourth, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] !=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            plt.clf()

            # create fig
            fig = plt.figure(figsize = (20,20), constrained_layout=False)
            gs = fig.add_gridspec(nrows=3, ncols=1)
            ax0 = fig.add_subplot(gs[:-1])
            ax1 = fig.add_subplot(gs[-1])

            # plot each quad
            ax0.plot(bins[:-1], count_first, 'o', color = 'black', label='Both approve')
            ax0.plot(bins[:-1], count_second, 'o', color = 'red', label=lbl1)
            ax0.plot(bins[:-1], count_third, 'o', color = 'grey', label='Both reject')
            ax0.plot(bins[:-1], count_fourth, 'o', color = 'blue', label=lbl2)
            ax0.set(ylabel='Count [$log_{10}$]')
            ax0.set_yscale('log')
            ax0.legend(frameon=True, framealpha=True)
            ax0.grid()

            # plot disagreement
            ax1.plot(bins[1], count_second[1]/count_total[1]*100, 'o', color = (0,0,0,0))
            ax1.plot(bins[:-1], count_second/count_total*100, 'o', color = 'red', label = lbl1)
            ax1.plot(bins[:-1], count_fourth/count_total*100, 'o', color = 'blue', label = lbl2)
            ax1.legend(['%s_%s_%s_nod0' %(step, chain.split('_')[1], chain.split('_')[2])], loc='best')
            ax1.set(ylabel='Disagreement [%]', xlabel='$E_{ratio}$')

            # save fig
            part = 'electron' if electron else 'jet'
            print('Saving ' + plot_path + '/quad_eratio_%s_%s_%s_%s ...'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_eratio_%s_%s_%s_%s.png'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_eratio_%s_%s_%s_%s.pdf'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.show()


# ### Quads for $f_{1}$

# In[30]:


def quad_f1(step_num=0, electron=True):
    var = 'trig_L2_cl_f1'

    for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][step_num]]:
        plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/quad/f1'

        chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                      'e26_lhtight_nod0_{RINGER}_ivarloose',
                      'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                      'e140_lhloose_nod0_{RINGER}'
                     ]

        bins = int(np.sqrt(table.loc[table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)]].shape[0]))

        for chain in chain_list:
            chain = step + '_' + chain
            print('Processing %s ...' %(chain))

            # create quadrant tables
            #first_quad = table.loc[(table[chain.format(RINGER = alg1)]==1) & (table[chain.format(RINGER = alg2)]==1)]['el_eta']
            #second_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] ==1)]['el_eta']
            #third_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']
            #fourth_quad = table.loc[(table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']

            # calculate counts for each quad
            ## constructing the tables here helps in saving memory
            upper_limit = 0.7
            lower_limit = -0.01
            
            [count_total, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_first, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)]==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_second, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)]==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_third, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)]!=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_fourth, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)]!=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            plt.clf()

            # create fig
            fig = plt.figure(figsize = (20,20), constrained_layout=False)
            gs = fig.add_gridspec(nrows=3, ncols=1)
            ax0 = fig.add_subplot(gs[:-1])
            ax1 = fig.add_subplot(gs[-1])

            # plot each quad
            ax0.plot(bins[:-1], count_first, 'o', color = 'black', label='Both approve')
            ax0.plot(bins[:-1], count_second, 'o', color = 'red', label=lbl1)
            ax0.plot(bins[:-1], count_third, 'o', color = 'grey', label='Both reject')
            ax0.plot(bins[:-1], count_fourth, 'o', color = 'blue', label=lbl2)
            ax0.set(ylabel='Count [$log_{10}$]')
            ax0.set_yscale('log')
            ax0.legend(frameon=True, framealpha=True)
            ax0.grid()

            # plot disagreement
            ax1.plot(bins[1], count_second[1]/count_total[1]*100, 'o', color = (0,0,0,0))
            ax1.plot(bins[:-1], count_second/count_total*100, 'o', color = 'red', label = lbl1)
            ax1.plot(bins[:-1], count_fourth/count_total*100, 'o', color = 'blue', label = lbl2)
            ax1.legend(['%s_%s_%s_nod0' %(step, chain.split('_')[1], chain.split('_')[2])], loc='best')
            ax1.set(ylabel='Disagreement [%]', xlabel='$f_{1}$')

            # save fig
            part = 'electron' if electron else 'jet'
            print('Saving ' + plot_path + '/quad_f1_%s_%s_%s_%s ...'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_f1_%s_%s_%s_%s.png'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_f1_%s_%s_%s_%s.pdf'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.show()


# ### Quads for $f_{3}$

# In[31]:


def quad_f3(step_num=0, electron=True):
    var = 'trig_L2_cl_f3'

    for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][step_num]]:
        plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/quad/f3'

        chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                      'e26_lhtight_nod0_{RINGER}_ivarloose',
                      'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                      'e140_lhloose_nod0_{RINGER}'
                     ]

        bins = int(np.sqrt(table.loc[table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)]].shape[0]))

        for chain in chain_list:
            chain = step + '_' + chain
            print('Processing %s ...' %(chain))

            # create quadrant tables
            #first_quad = table.loc[(table[chain.format(RINGER = alg1)]==1) & (table[chain.format(RINGER = alg2)]==1)]['el_eta']
            #second_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] ==1)]['el_eta']
            #third_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']
            #fourth_quad = table.loc[(table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']

            # calculate counts for each quad
            ## constructing the tables here helps in saving memory
            upper_limit = 0.15
            lower_limit = -0.05
            
            [count_total, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_first, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)]==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_second, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)]==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_third, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)]!=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_fourth, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)]!=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            plt.clf()

            # create fig
            fig = plt.figure(figsize = (20,20), constrained_layout=False)
            gs = fig.add_gridspec(nrows=3, ncols=1)
            ax0 = fig.add_subplot(gs[:-1])
            ax1 = fig.add_subplot(gs[-1])

            # plot each quad
            ax0.plot(bins[:-1], count_first, 'o', color = 'black', label='Both approve')
            ax0.plot(bins[:-1], count_second, 'o', color = 'red', label=lbl1)
            ax0.plot(bins[:-1], count_third, 'o', color = 'grey', label='Both reject')
            ax0.plot(bins[:-1], count_fourth, 'o', color = 'blue', label=lbl2)
            ax0.set(ylabel='Count [$log_{10}$]')
            ax0.set_yscale('log')
            ax0.legend(frameon=True, framealpha=True)
            ax0.grid()

            # plot disagreement
            ax1.plot(bins[1], count_second[1]/count_total[1]*100, 'o', color = (0,0,0,0))
            ax1.plot(bins[:-1], count_second/count_total*100, 'o', color = 'red', label = lbl1)
            ax1.plot(bins[:-1], count_fourth/count_total*100, 'o', color = 'blue', label = lbl2)
            ax1.legend(['%s_%s_%s_nod0' %(step, chain.split('_')[1], chain.split('_')[2])], loc='best')
            ax1.set(ylabel='Disagreement [%]', xlabel='$f_{3}$')

            # save fig
            part = 'electron' if electron else 'jet'
            print('Saving ' + plot_path + '/quad_f3_%s_%s_%s_%s ...'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_f3_%s_%s_%s_%s.png'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_f3_%s_%s_%s_%s.pdf'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.show()


# ### Quads for $w_{\eta}$

# In[32]:


def quad_weta(step_num=0, electron=True):
    var = 'trig_L2_cl_weta2'

    for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][step_num]]:
        plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/quad/weta'

        chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                      'e26_lhtight_nod0_{RINGER}_ivarloose',
                      'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                      'e140_lhloose_nod0_{RINGER}'
                     ]

        bins = int(np.sqrt(table.loc[table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)]].shape[0]))

        for chain in chain_list:
            chain = step + '_' + chain
            print('Processing %s ...' %(chain))

            # create quadrant tables
            #first_quad = table.loc[(table[chain.format(RINGER = alg1)]==1) & (table[chain.format(RINGER = alg2)]==1)]['el_eta']
            #second_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] ==1)]['el_eta']
            #third_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']
            #fourth_quad = table.loc[(table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']

            # calculate counts for each quad
            ## constructing the tables here helps in saving memory
            upper_limit = 0.02
            lower_limit = 0.005
            
            [count_total, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_first, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)]==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_second, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)]==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_third, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)]!=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_fourth, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)]!=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            plt.clf()

            # create fig
            fig = plt.figure(figsize = (20,20), constrained_layout=False)
            gs = fig.add_gridspec(nrows=3, ncols=1)
            ax0 = fig.add_subplot(gs[:-1])
            ax1 = fig.add_subplot(gs[-1])

            # plot each quad
            ax0.plot(bins[:-1], count_first, 'o', color = 'black', label='Both approve')
            ax0.plot(bins[:-1], count_second, 'o', color = 'red', label=lbl1)
            ax0.plot(bins[:-1], count_third, 'o', color = 'grey', label='Both reject')
            ax0.plot(bins[:-1], count_fourth, 'o', color = 'blue', label=lbl2)
            ax0.set(ylabel='Count [$log_{10}$]')
            ax0.set_yscale('log')
            ax0.legend(frameon=True, framealpha=True)
            ax0.grid()

            # plot disagreement
            ax1.plot(bins[1], count_second[1]/count_total[1]*100, 'o', color = (0,0,0,0))
            ax1.plot(bins[:-1], count_second/count_total*100, 'o', color = 'red', label = lbl1)
            ax1.plot(bins[:-1], count_fourth/count_total*100, 'o', color = 'blue', label = lbl2)
            ax1.legend(['%s_%s_%s_nod0' %(step, chain.split('_')[1], chain.split('_')[2])], loc='best')
            ax1.set(ylabel='Disagreement [%]', xlabel='$w_{\eta}$')

            # save fig
            part = 'electron' if electron else 'jet'
            print('Saving ' + plot_path + '/quad_weta_%s_%s_%s_%s ...'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_weta_%s_%s_%s_%s.png'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_weta_%s_%s_%s_%s.pdf'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.show()


# ### Quads for $w_{stot}$

# In[33]:


def quad_wstot(step_num=0, electron=True):
    var = 'trig_L2_cl_wstot'

    for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][step_num]]:
        plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/quad/wstot'

        chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                      'e26_lhtight_nod0_{RINGER}_ivarloose',
                      'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                      'e140_lhloose_nod0_{RINGER}'
                     ]

        bins = int(np.sqrt(table.loc[table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)]].shape[0]))

        for chain in chain_list:
            chain = step + '_' + chain
            print('Processing %s ...' %(chain))

            # create quadrant tables
            #first_quad = table.loc[(table[chain.format(RINGER = alg1)]==1) & (table[chain.format(RINGER = alg2)]==1)]['el_eta']
            #second_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] ==1)]['el_eta']
            #third_quad = table.loc[(table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']
            #fourth_quad = table.loc[(table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)] !=1)]['el_eta']

            # calculate counts for each quad
            ## constructing the tables here helps in saving memory
            upper_limit = 8
            lower_limit = 0
            
            [count_total, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_first, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)]==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_second, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)]==1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_third, bins, nome]  = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] !=1) & (table[chain.format(RINGER = alg2)]!=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            [count_fourth, bins, nome] = plt.hist(table.loc[(table['target']==electron) & (table[chain.format(RINGER = alg1)] ==1) & (table[chain.format(RINGER = alg2)]!=1) & (table[var]<=upper_limit) & (table[var]>=lower_limit)][var], bins=bins)
            plt.clf()

            # create fig
            fig = plt.figure(figsize = (20,20), constrained_layout=False)
            gs = fig.add_gridspec(nrows=3, ncols=1)
            ax0 = fig.add_subplot(gs[:-1])
            ax1 = fig.add_subplot(gs[-1])

            # plot each quad
            ax0.plot(bins[:-1], count_first, 'o', color = 'black', label='Both approve')
            ax0.plot(bins[:-1], count_second, 'o', color = 'red', label=lbl1)
            ax0.plot(bins[:-1], count_third, 'o', color = 'grey', label='Both reject')
            ax0.plot(bins[:-1], count_fourth, 'o', color = 'blue', label=lbl2)
            ax0.set(ylabel='Count [$log_{10}$]')
            ax0.set_yscale('log')
            ax0.legend(frameon=True, framealpha=True)
            ax0.grid()

            # plot disagreement
            ax1.plot(bins[1], count_second[1]/count_total[1]*100, 'o', color = (0,0,0,0))
            ax1.plot(bins[:-1], count_second/count_total*100, 'o', color = 'red', label = lbl1)
            ax1.plot(bins[:-1], count_fourth/count_total*100, 'o', color = 'blue', label = lbl2)
            ax1.legend(['%s_%s_%s_nod0' %(step, chain.split('_')[1], chain.split('_')[2])], loc='best')
            ax1.set(ylabel='Disagreement [%]', xlabel='$w_{stot}$')

            # save fig
            part = 'electron' if electron else 'jet'
            print('Saving ' + plot_path + '/quad_wstot_%s_%s_%s_%s ...'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_wstot_%s_%s_%s_%s.png'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.savefig(plot_path + '/quad_wstot_%s_%s_%s_%s.pdf'%(part, file_name, chain.split('_')[0], chain.split('_')[1]))
            fig.show()


# In[34]:


from itertools import product
# Algorithms to plot and their names
algorithms = {'noringer': 'NoRinger reject',
              'ringer_v8_new':'V8 reject',
              'ringer_v8_half': 'V12 reject',
              'ringer_v8_14': 'V8.14 reject',
             }
done = set()

for comparison in product(algorithms, repeat=2):
    # Steps to ensure no repetition
    if comparison[0] == comparison[1]:
        continue
    if comparison in done:
        continue
    for i in product(comparison, repeat=2):
        done.add(i)

    # Setting up
    alg1, alg2 = comparison
    lbl1, lbl2 = algorithms[alg1], algorithms[alg2]
    file_name = lbl1.split(' ')[0] + 'x' + lbl2.split(' ')[0]
    
    # Variables to plot
    for i in tqdm(range(3+1)):
        quad_reta(i)
        quad_eratio(i)
        quad_f1(i)
        quad_f3(i)
        quad_weta(i)
        quad_wstot(i)
    print('Done')


# In[ ]:


print('End of script')

