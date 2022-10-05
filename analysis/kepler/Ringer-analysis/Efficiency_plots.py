#!/usr/bin/env python
# coding: utf-8

# # Kepler Framework Examples

# In[1]:


from kepler.pandas.menu       import ElectronSequence as Chain
from kepler.pandas.readers    import load, load_in_loop
from kepler.pandas.decorators import create_ringer_v8_decorators, create_ringer_v9_decorators, RingerDecorator
from kepler.pandas.decorators import create_ringer_v8_new_decorators, create_ringer_v8_half_fast_decorators, create_ringer_v8_34_decorators, create_ringer_v8_half_decorators, create_ringer_v8_14_decorators

import kepler
import tqdm
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

# In[2]:


## codigo original
# dpath = '/home/jodafons/public/cern_data/new_files/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins'
# dpath+= '/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins_et{ET}_eta{ETA}.npz'
# paths = []
# for et in range(5):
#     for eta in range(5):
#         paths.append( dpath.format(ET=et,ETA=eta) )

real_run = False


# ## Load Data

# In[3]:


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


# In[4]:


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


# In[5]:


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

# In[6]:


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

# In[7]:


table = load_in_loop( paths, drop_columns=drop_columns, decorators=decorators, chains=chains )


# In[8]:


if True:
    print(table.columns.to_list())


# In[9]:


table.head()


# ## Efficiency plots

# ##### kWhite  = 0,   kBlack  = 1,   kGray    = 920,  kRed    = 632,  kGreen  = 416,
# ##### kBlue   = 600, kYellow = 400, kMagenta = 616,  kCyan   = 432,  kOrange = 800,
# ##### kSpring = 820, kTeal   = 840, kAzure   =  860, kViolet = 880,  kPink   = 900

# https://root.cern.ch/doc/master/classTColor.html

# In[13]:


def hist1d( name, values, bins, density=False ):
    counts, dummy = np.histogram(values, bins=bins, density=density )
    hist = ROOT.TH1F( name, '', len(bins)-1, array.array('d',bins))
    root_numpy.array2hist(counts, hist)
    return hist

colors  = [ROOT.kBlack, ROOT.kBlue+1, ROOT.kGreen+1, ROOT.kRed+1] # Primarias
#colors  = [ROOT.kBlue-4, ROOT.kBlack, ROOT.kGreen-4, ROOT.kGray] # cores do Mica
#markers = [22, 26, 23, 32]
markers = [33, 22, 23, 30]

def add_legend(x, y, legends):
    rpl.add_legend( legends, x, y, x+0.98, y+0.20, textsize=12, option='p' )


# ### Efficiency with respect to $E_T$

# In[11]:


def make_et_plot(dataframe, chain, chain_step, l2suffix, fake=False):
    var = 'trig_L2_cl_et'
    from Gaugi.constants import GeV
    # plot in eta need sum 1 in chain threshold 

    m_bins = [4,7,10,15,20,25,30,35,40,45,50,60,80,150,300] # et_bins
    #m_bins = [15, 20, 30, 40, 50, 1000000]
    #m_bins = np.arange(3, 16, step=.5).tolist()
    et_cut  = int(chain.split('_')[1][1:])
    offline = chain.split('_')[2]
    if fake:
        #aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_lhvloose != 1) &\
        aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                *GeV
    else:
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                                                       *GeV
    
    #step_decision = chain_step + '_' + '_'.join(chain.split('_')[1:])
    step_decision = chain
    #'L2Calo_e26_lhtight_nod0_ringer_v8_34_ivarloose'
    # cuts for all
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('et_num', passed[var]/GeV, m_bins )
    #                                       /GeV
    h_den = hist1d('et_den', total[var]/GeV, m_bins )
    #                                      /GeV
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)


# In[12]:


for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][3]]:
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/effs/et'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_et_plot(table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', 
                        ),
            # no ringer 2017
            make_et_plot(table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', 
                        ),
            # no ringer 2017
            make_et_plot(table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', 
                        ),
            ## no ringer 2017
            #make_et_plot(table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', 
            #            ),
            # no ringer 2017
            make_et_plot(table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', 
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], 'E_{T} [GeV]', colors, markers)
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/eff_et_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_et_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_et_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# In[13]:


for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][3]]:
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/frs/et'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_et_plot(table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', fake = True
                        ),
            # no ringer 2017
            make_et_plot(table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', fake = True
                        ),
            # no ringer 2017
            make_et_plot(table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', fake = True
                        ),
            ## no ringer 2017
            #make_et_plot(table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', fake = True
            #            ),
            # no ringer 2017
            make_et_plot(table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', fake = True
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], 'E_{T} [GeV]', colors, markers, ylabel='Fake Rate')
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - F_{R} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/fr_et_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/fr_et_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/fr_et_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# ### Efficiency with respect to $\eta$

# In[14]:


def make_eta_plot(dataframe, chain, chain_step, l2suffix, fake=False):
    var = 'trig_L2_cl_eta'
    from Gaugi.constants import GeV
    # plot in eta need sum 1 in chain threshold 

    m_bins = [-2.47,-2.37,-2.01,-1.81,-1.52,-1.37,-1.15,-0.80,-0.60,-0.10,0.00,
              0.10, 0.60, 0.80, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]
    
    et_cut  = int(chain.split('_')[1][1:])
    offline = chain.split('_')[2]
    if fake:
        #aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_lhvloose != 1) &\
        aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                *GeV
    else:
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                                                       *GeV
    
    #step_decision = chain_step + '_' + '_'.join(chain.split('_')[1:])
    step_decision = chain
    #'L2Calo_e26_lhtight_nod0_ringer_v8_34_ivarloose'
    # cuts for all
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('eta_num', passed[var], m_bins )
    h_den = hist1d('eta_den', total[var], m_bins )
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)


# In[15]:


for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][3]]:
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/effs/eta'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_eta_plot(table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', 
                        ),
            # no ringer 2017
            make_eta_plot(table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', 
                        ),
            # no ringer 2017
            make_eta_plot(table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', 
                        ),
            ## no ringer 2017
            #make_eta_plot(table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', 
            #            ),
            # no ringer 2017
            make_eta_plot(table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', 
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], '#eta', colors, markers)
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1)
        fig.show()
        print('Saving '+ plot_path+'/eff_eta_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_eta_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_eta_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# In[16]:


for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][3]]:
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/frs/eta'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_eta_plot(table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', fake = True
                        ),
            # no ringer 2017
            make_eta_plot(table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', fake = True
                        ),
            # no ringer 2017
            make_eta_plot(table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', fake = True
                        ),
            ## no ringer 2017
            #make_eta_plot(table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', fake = True
            #            ),
            # no ringer 2017
            make_eta_plot(table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', fake = True
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], '#eta', colors, markers, ylabel='Fake Rate')
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - F_{R} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1)
        fig.show()
        print('Saving '+ plot_path+'/fr_eta_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/fr_eta_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/fr_eta_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# ### Efficiency with respect to $pT$

# In[17]:


def make_pt_plot(dataframe, chain, chain_step, l2suffix, fake=False):
    var = 'trig_L2_el_pt'
    from Gaugi.constants import GeV, MeV
    # plot in eta need sum 1 in chain threshold 

    #m_bins = [4,7,10,15,20,25,30,35,40,45,50,60,80,150,300] # et_bins
    #m_bins = [15, 20, 30, 40, 50, 1000000]
    m_bins = np.arange(0, 2000*10**3//2, step=50*10**3).tolist()
    et_cut  = int(chain.split('_')[1][1:])
    if fake:
        #aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_lhvloose != 1) &\
        aux_df = dataframe.loc[(dataframe.target != 1) &                               (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                      *GeV
    else:
        offline = chain.split('_')[2]
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                                                       *GeV
    
    #step_decision = chain_step + '_' + '_'.join(chain.split('_')[1:])
    step_decision = chain
    #'L2Calo_e26_lhtight_nod0_ringer_v8_34_ivarloose'
    # cuts for all
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('pt_num', passed[var]/MeV, m_bins )
    #                                               /GeV
    h_den = hist1d('pt_den', total[var]/MeV, m_bins )
    #                                              /GeV
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)


# In[18]:


for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][3]]:
#step = 'L2'
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/effs/pt'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_pt_plot(table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', 
                        ),
            # no ringer 2017
            make_pt_plot(table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', 
                        ),
            # no ringer 2017
            make_pt_plot(table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', 
                        ),
            # no ringer 2017
            #make_pt_plot(table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', 
            #            ),
            # no ringer 2017
            make_pt_plot(table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', 
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], 'pT [MeV]', colors, markers)
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.15,0.15, labels)
        rpl.add_text( 0.15, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=0, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/eff_pt_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_pt_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_pt_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# In[19]:


for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][3]]:
    #step = 'L2'
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/frs/pt'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_pt_plot(table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', fake=True
                        ),
            # no ringer 2017
            make_pt_plot(table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', fake=True
                        ),
            # no ringer 2017
            make_pt_plot(table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', fake=True
                        ),
            # no ringer 2017
            #make_pt_plot(table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', fake=True
            #            ),
            # no ringer 2017
            make_pt_plot(table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', fake=True
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], 'pT [MeV]', colors, markers, ylabel = 'Fake Rate')
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - F_{R} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.15,0.6, labels)
        rpl.add_text( 0.15, 0.8, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=0, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/fr_pt_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/fr_pt_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/fr_pt_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# ### Efficiency with respect to $< \mu >$

# In[20]:


def make_mu_plot(dataframe, chain, chain_step, l2suffix, fake=False):
    var = 'avgmu'
    from Gaugi.constants import GeV, MeV
    # plot in eta need sum 1 in chain threshold 

    m_bins = [10, 20, 30, 40, 50, 60, 70] # et_bins
    #m_bins = [15, 20, 30, 40, 50, 1000000]
    #m_bins = np.arange(0, 2000*10**3, step=50*10**3).tolist()
    et_cut  = int(chain.split('_')[1][1:])
    if fake:
        #aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_lhvloose != 1) &\
        aux_df = dataframe.loc[(dataframe.target != 1) &                               (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                      *GeV
    else:
        offline = chain.split('_')[2]
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                                                       *GeV
    
    #step_decision = chain_step + '_' + '_'.join(chain.split('_')[1:])
    step_decision = chain
    #'L2Calo_e26_lhtight_nod0_ringer_v8_34_ivarloose'
    # cuts for all
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('mu_num', passed[var]/MeV, m_bins )
    #                                               /GeV
    h_den = hist1d('mu_den', total[var]/MeV, m_bins )
    #                                              /GeV
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)


# In[21]:


for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][3]]:
#step = 'L2'
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/effs/mu'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_mu_plot(table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', 
                        ),
            # Ringer V8
            make_mu_plot(table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', 
                        ),
            # Ringer V8.1/2
            make_mu_plot(table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', 
                        ),
            ## Ringer V8.3/4
            #make_mu_plot(table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', 
            #            ),
            # Ringer V8.3/4
            make_mu_plot(table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', 
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], '< #mu >', colors, markers)
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/eff_mu_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_mu_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_mu_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# In[22]:


for step in [['L2Calo', 'L2', 'EFCalo', 'HLT'][3]]:
#step = 'L2'
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/frs/mu'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_mu_plot(table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', fake=True
                        ),
            # Ringer V8
            make_mu_plot(table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', fake=True
                        ),
            # Ringer V8.1/2
            make_mu_plot(table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', fake=True
                        ),
            ## Ringer V8.3/4
            #make_mu_plot(table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', fake=True
            #            ),
            # Ringer V8.3/4
            make_mu_plot(table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', fake=True
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], '< #mu >', colors, markers, ylabel = 'Fake Rate')
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/eff_mu_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_mu_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_mu_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# # Running for Boosted

# ## Load Data

# In[4]:


# codigo modificado
boosted_dpath = '/home/pedro.lima/data/mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2'
boosted_dpath+= '/mc16_13TeV.302236_309995_341330.sgn.boosted_probes.WZ_llqq_plus_radion_ZZ_llqq_plus_ggH3000.merge.25bins.v2_et{ET}_eta{ETA}.npz'
boosted_paths = []

#for et in [4]:
#    for eta in [0]:
#        boosted_paths.append( boosted_dpath.format(ET=et,ETA=eta) )
for et in range(5):
    for eta in range(5):
        boosted_paths.append( boosted_dpath.format(ET=et,ETA=eta) )


# In[5]:


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

#drop_columns.extend( ['trig_L2_cl_ring_%d'%i for i in range(100)] )


# In[6]:


os.environ['RINGER_TUNING_PATH']='/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/tunings'

decorators = create_ringer_v8_new_decorators()
decorators += create_ringer_v8_half_fast_decorators()
#decorators+= create_ringer_v9_decorators()
#decorators += create_ringer_v8_34_decorators()
decorators += create_ringer_v8_14_decorators()


# ## Setup Chains

# In[7]:


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

# In[8]:


boosted_table = load_in_loop( boosted_paths, drop_columns=drop_columns, decorators=decorators, chains=chains )


# In[22]:


if True:
    print(boosted_table.columns.to_list())


# ## Efficiency plots

# ### Efficiency with respect to $E_T$

# In[23]:


def make_et_plot(dataframe, chain, chain_step, l2suffix, fake=False):
    var = 'trig_L2_cl_et'
    from Gaugi.constants import GeV
    # plot in eta need sum 1 in chain threshold 

    m_bins = [4,7,10,15,20,25,30,35,40,45,50,60,80,150,300] # et_bins
    #m_bins = [15, 20, 30, 40, 50, 1000000]
    #m_bins = np.arange(3, 16, step=.5).tolist()
    et_cut  = int(chain.split('_')[1][1:])
    if fake:
        #aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_lhvloose != 1) &\
        aux_df = dataframe.loc[(dataframe.target != 1) &                               (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                      *GeV
    else:
        offline = chain.split('_')[2]
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                                                       *GeV
    
    #step_decision = chain_step + '_' + '_'.join(chain.split('_')[1:])
    step_decision = chain
    #'L2Calo_e26_lhtight_nod0_ringer_v8_34_ivarloose'
    # cuts for all
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('et_b_num', passed[var]/GeV, m_bins )
    #                                         /GeV
    h_den = hist1d('et_b_den', total[var]/GeV, m_bins )
    #                                        /GeV
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)


# In[24]:


for step in ['L2Calo', 'L2', 'EFCalo', 'HLT']:
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/effs/et'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_et_plot(boosted_table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', 
                        ),
            # no ringer 2017
            make_et_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', 
                        ),
            # no ringer 2017
            make_et_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', 
                        ),
            ## no ringer 2017
            #make_et_plot(boosted_table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', 
            #            ),
            # no ringer 2017
            make_et_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', 
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], 'E_{T} [GeV]', colors, markers)
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text(0.55, 0.4, 'Boosted', textsize=0.04)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/eff_et_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_boosted_et_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_boosted_et_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# ### Efficiency with respect to $\Delta R$

# In[25]:


def make_dr_plot(dataframe, chain, chain_step, l2suffix, fake=False):
    var = 'el_TaP_deltaR'
    from Gaugi.constants import GeV
    # plot in eta need sum 1 in chain threshold 

    m_bins = [0, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.22, 0.24, 0.26, 0.28, 0.3, 0.35, 0.40, 0.6] # et_bins
    #m_bins = [15, 20, 30, 40, 50, 1000000]
    #_bins = np.arange(0, 1, step=.05).tolist()
    et_cut  = int(chain.split('_')[1][1:])
    if fake:
        #aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_lhvloose != 1) &\
        aux_df = dataframe.loc[(dataframe.target != 1) &                               (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                      *GeV
    else:
        offline = chain.split('_')[2]
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                                                       *GeV
    
    #step_decision = chain_step + '_' + '_'.join(chain.split('_')[1:])
    step_decision = chain
    #'L2Calo_e26_lhtight_nod0_ringer_v8_34_ivarloose'
    # cuts for all
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('dr_b_num', passed[var], m_bins )
    #                                               /GeV
    h_den = hist1d('dr_b_den', total[var], m_bins )
    #                                              /GeV
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)


# In[26]:


for step in ['L2Calo', 'L2', 'EFCalo', 'HLT']:
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/effs/deltar'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_dr_plot(boosted_table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', 
                        ),
            # no ringer 2017
            make_dr_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', 
                        ),
            # no ringer 2017
            make_dr_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', 
                        ),
            ## no ringer 2017
            #make_dr_plot(boosted_table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', 
            #            ),
            # no ringer 2017
            make_dr_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', 
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], '\Delta R', colors, markers)
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text(0.55, 0.4, 'Boosted', textsize=0.04)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=-0.5, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/eff_boosted_deltaR_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_boosted_deltaR_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_boosted_deltaR_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# ### Efficiency with respect to $pT$

# In[27]:


def make_pt_plot(dataframe, chain, chain_step, l2suffix, fake=False):
    var = 'trig_L2_el_pt'
    from Gaugi.constants import GeV, MeV
    # plot in eta need sum 1 in chain threshold 

    #m_bins = [4,7,10,15,20,25,30,35,40,45,50,60,80,150,300] # et_bins
    #m_bins = [15, 20, 30, 40, 50, 1000000]
    m_bins = np.arange(0, 2000*10**3, step=50*10**3).tolist()
    et_cut  = int(chain.split('_')[1][1:])
    if fake:
        #aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_lhvloose != 1) &\
        aux_df = dataframe.loc[(dataframe.target != 1) &                               (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                      *GeV
    else:
        offline = chain.split('_')[2]
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                                                       *GeV
    
    #step_decision = chain_step + '_' + '_'.join(chain.split('_')[1:])
    step_decision = chain
    #'L2Calo_e26_lhtight_nod0_ringer_v8_34_ivarloose'
    # cuts for all
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('pt_b_num', passed[var]/MeV, m_bins )
    #                                               /GeV
    h_den = hist1d('pt_b_den', total[var]/MeV, m_bins )
    #                                              /GeV
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)


# In[ ]:


for step in ['L2Calo', 'L2', 'EFCalo', 'HLT']:
#step = 'L2'
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/effs/pt'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_pt_plot(boosted_table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', 
                        ),
            # no ringer 2017
            make_pt_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', 
                        ),
            # no ringer 2017
            make_pt_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', 
                        ),
            ## no ringer 2017
            #make_pt_plot(boosted_table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', 
            #            ),
            # no ringer 2017
            make_pt_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', 
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], 'pT [MeV]', colors, markers)
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text(0.55, 0.4, 'Boosted', textsize=0.04)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=0, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/eff_boosted_pt_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_boosted_pt_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_boosted_pt_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# ### Efficiency with respect to $< \mu >$

# In[ ]:


def make_mu_plot(dataframe, chain, chain_step, l2suffix, fake=False):
    var = 'avgmu'
    from Gaugi.constants import GeV, MeV
    # plot in eta need sum 1 in chain threshold 

    #m_bins = [4,7,10,15,20,25,30,35,40,45,50,60,80,150,300] # et_bins
    #m_bins = [15, 20, 30, 40, 50, 1000000]
    #m_bins = [10, 20, 30, 40, 50, 60, 70]
    m_bins = [0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    et_cut  = int(chain.split('_')[1][1:])
    if fake:
        #aux_df = dataframe.loc[(dataframe.target != 1) & (dataframe.el_lhvloose != 1) &\
        aux_df = dataframe.loc[(dataframe.target != 1) &                               (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                      *GeV
    else:
        offline = chain.split('_')[2]
        aux_df = dataframe.loc[(dataframe.target == 1) & (dataframe['el_%s' %(offline)] == 1) & (dataframe.el_et >= (et_cut - 5)*GeV) & (np.abs(dataframe.el_eta) <=2.47)]
        #                                                                                                                       *GeV
    
    #step_decision = chain_step + '_' + '_'.join(chain.split('_')[1:])
    step_decision = chain
    #'L2Calo_e26_lhtight_nod0_ringer_v8_34_ivarloose'
    # cuts for all
    total   = aux_df
    passed  = aux_df.loc[(aux_df[step_decision] == 1)]
    
    h_num = hist1d('mu_b_num', passed[var]/MeV, m_bins )
    #                                               /GeV
    h_den = hist1d('mu_b_den', total[var]/MeV, m_bins )
    #                                              /GeV
    h_eff = rpl.hist1d.divide(h_num,h_den) 
    
    return h_eff, len(passed)/len(total)


# In[ ]:


for step in ['L2Calo', 'L2', 'EFCalo', 'HLT']:
#step = 'L2'
    plot_path = '/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/kepler/kepler_imgs/effs/mu'

    chain_list = ['e24_lhtight_nod0_{RINGER}_ivarloose',
                  'e26_lhtight_nod0_{RINGER}_ivarloose',
                  'e60_lhmedium_nod0_{RINGER}_L1EM24VHI',
                  'e140_lhloose_nod0_{RINGER}'
                 ]

    for chain in chain_list:

        ringer_list = ['noringer', 
                       'ringer_v8_new', 
                       'ringer_v8_half',
                       #'ringer_v8_34',
                       'ringer_v8_14'
                      ]

        label_list = ['NoRinger', 
                      'Ringer V8', 
                      'Ringer V8.1/2',
                      #'Ringer V8.3/4',
                      'Ringer V8.1/4'
                     ]


        #chain = 'e26_lh{OP}_nod0_{RINGER}_ivarloose'
        #e26_lh{OP}_nod0_{RINGER}_ivarloose


        trigger = step+'_'+chain

        m_info = np.array([
            # ringer 2017
            make_mu_plot(boosted_table, 
                         trigger.format(RINGER='noringer'), chain_step=step, l2suffix='noringer', 
                        ),
            # no ringer 2017
            make_mu_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_new'), chain_step=step, l2suffix='ringer_v8_new', 
                        ),
            # no ringer 2017
            make_mu_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_half'), chain_step=step, l2suffix='ringer_v8_half', 
                        ),
            ## no ringer 2017
            #make_mu_plot(boosted_table, 
            #             trigger.format(RINGER='ringer_v8_34'), chain_step=step, l2suffix='ringer_v8_34', 
            #            ),
            # no ringer 2017
            make_mu_plot(boosted_table, 
                         trigger.format(RINGER='ringer_v8_14'), chain_step=step, l2suffix='ringer_v8_14', 
                        )
        ])

        # make the plot
        fig = rpl.create_canvas('my_canvas', canw=1400, canh=1000)
        fig = rpl.plot_profiles( m_info[:,0], '< #mu >', colors, markers)
        #rpl.set_atlas_label(0.15,0.85,'Internal, pp data #sqrt{s}= 13TeV')
        rpl.format_canvas_axes(YTitleOffset = 0.95)
        labels = []
        for idx, ilabel in enumerate(label_list):
            labels.append('%s - P_{D} (%s): %1.2f %%' %(ilabel, step, m_info[idx, 1]*100))
        add_legend( 0.55,0.15, labels)
        rpl.add_text(0.55, 0.4, 'Boosted', textsize=0.04)
        rpl.add_text( 0.55, 0.35, '%s_%s_%s_nod0' %(step, chain.split('_')[0], chain.split('_')[1]), textsize=0.04)
        rpl.fix_yaxis_ranges( ignore_zeros=True, ignore_errors=True , yminf=0, ymaxf=1.1) 
        fig.show()
        print('Saving '+ plot_path+'/eff_boosted_mu_%s_%s_%s_root' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_boosted_mu_%s_%s_%s_root.pdf' %(step, chain.split('_')[0], chain.split('_')[1]))
        fig.savefig(plot_path+'/eff_boosted_mu_%s_%s_%s_root.png' %(step, chain.split('_')[0], chain.split('_')[1]))


# In[ ]:


print('End of script')


# In[ ]:




