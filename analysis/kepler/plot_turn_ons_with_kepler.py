#!/usr/bin/env python
# coding: utf-8

# # Plot turn on curve with kepler

# In[ ]:


from kepler.pandas import RingerDecorator
from kepler.pandas import ElectronSequence as Chain
from kepler.pandas import load
from kepler.pandas import drop_ring_columns
import numpy as np
import pandas as pd
import collections
import os
import matplotlib
import matplotlib.pyplot as plt
from pprint import pprint
from copy import deepcopy
import gc

import mplhep as hep
plt.style.use(hep.style.ROOT)

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[1]:


# data path
dpath = '/home/pedro.lima/data/new_files/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins'
dpath+= '/data17_13TeV.AllPeriods.sgn.probes_lhvloose_EGAM1.bkg.vprobes_vlhvloose_EGAM7.GRL_v97.25bins_et{ET}_eta{ETA}.npz'


# In[2]:


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
       'trig_L2_el_pt', 
       'trig_L2_el_eta',
       'trig_L2_el_phi', 
       'trig_L2_el_caloEta', 
       'trig_L2_el_trkClusDeta',
       'trig_L2_el_trkClusDphi', 
       'trig_L2_el_etOverPt', 
       'trig_EF_cl_et',
       #'trig_EF_el_et', 
       #'trig_EF_el_lhtight', 
       #'trig_EF_el_lhmedium',
       #'trig_EF_el_lhloose', 
       #'trig_EF_el_lhvloose', 
       #'el_et', 
       #'el_eta',
       'el_etaBE2', 
       #'el_phi', 
       #'el_rhad1', 
       #'el_rhad', 
       #'el_f3', 
       #'el_weta2',
       #'el_rphi',
       #'el_reta', 
       #'el_wtots1', 
       #'el_eratio', 
       #'el_f1', 
       #'el_hastrack',
       #'el_numberOfBLayerHits', 
       #'el_numberOfPixelHits', 
       #'el_numberOfTRTHits',
       #'el_d0',
       #'el_d0significance', 
       #'el_eProbabilityHT', 
       #'el_trans_TRT_PID',
       #'el_deltaEta1', 
       #'el_deltaPhi2', 
       #'el_deltaPhi2Rescaled',
       #'el_deltaPOverP', 
       #'el_lhtight',
       #'el_lhmedium',
       #'el_lhloose',
       #'el_lhvloose', 
       #'el_TaP_deltaR', 
       #'trig_EF_el_lhtight_ivarloose',
       'L1_EM3', 
       'L1_EM7', 
       'L1_EM15VH', 
       'L1_EM15VHI',
       'L1_EM20VH',
       'L1_EM20VHI', 
       'L1_EM22VH', 
       'L1_EM22VHI', 
       'L1_EM24VHI',
       'trig_L2_cl_lhvloose_et0to12', 
       'trig_L2_cl_lhvloose_et12to20',
       'trig_L2_cl_lhvloose_et22toInf', 
       'trig_L2_cl_lhloose_et0to12',
       'trig_L2_cl_lhloose_et12to20', 
       'trig_L2_cl_lhloose_et22toInf',
       'trig_L2_cl_lhmedium_et0to12', 
       'trig_L2_cl_lhmedium_et12to20',
       'trig_L2_cl_lhmedium_et22toInf', 
       'trig_L2_cl_lhtight_et0to12',
       'trig_L2_cl_lhtight_et12to20', 
       'trig_L2_cl_lhtight_et22toInf',
       'trig_L2_el_cut_pt0to15', 
       'trig_L2_el_cut_pt15to20',
       'trig_L2_el_cut_pt20to50', 
       'trig_L2_el_cut_pt50toInf', 
       #'target'
       ]


# ## Setup Ringer v8:

# In[ ]:


def my_generator( df ):
    col_names= ['trig_L2_cl_ring_%d'%i for i in range(100)]
    rings = df[col_names].values.astype(np.float32)
    def norm1( data ):
        norms = np.abs( data.sum(axis=1) )
        norms[norms==0] = 1
        return data/norms[:,None]
    rings = norm1(rings)
    return [rings]

# tuning path
ringer_version = "TrigL2_20180125_v8"

tuning_path = os.path.join('/home/pedro.lima/workspace/CERN-ATLAS-Qualify-1/analysis/tunings',ringer_version)

tight_path = tuning_path+'/ElectronRingerTightTriggerConfig.conf'
medium_path = tuning_path+'/ElectronRingerMediumTriggerConfig.conf'
loose_path = tuning_path+'/ElectronRingerLooseTriggerConfig.conf'
vloose_path = tuning_path+'/ElectronRingerVeryLooseTriggerConfig.conf'

v8_tight = RingerDecorator(tight_path, my_generator)
v8_medium = RingerDecorator(medium_path, my_generator)
v8_loose = RingerDecorator(loose_path, my_generator)
v8_vloose = RingerDecorator(vloose_path, my_generator)

# ringer_version = "TrigL2_20210907_v8-1"
# tuning_path = os.path.join('/home/natmourajr/Workspace/CERN/CERN-ATLAS-Qualify/analysis/tunings',ringer_version)

# tight_path = tuning_path+'/ElectronRingerTightTriggerConfig.conf'
# medium_path = tuning_path+'/ElectronRingerMediumTriggerConfig.conf'
# loose_path = tuning_path+'/ElectronRingerLooseTriggerConfig.conf'
# vloose_path = tuning_path+'/ElectronRingerVeryLooseTriggerConfig.conf'

# v8_half_tight = RingerDecorator(tight_path, my_generator)
# v8_half_medium = RingerDecorator(medium_path, my_generator)
# v8_half_loose = RingerDecorator(loose_path, my_generator)
# v8_half_vloose = RingerDecorator(vloose_path, my_generator)


decorators = [('ringer_v8_tight', v8_tight),
              ('ringer_v8_medium', v8_medium),
              ('ringer_v8_loose', v8_loose),
              ('ringer_v8_veryloose', v8_loose),
#               ('ringer_v8_1/2_tight', v8_half_tight),
#               ('ringer_v8_1/2_medium', v8_half_medium),
#               ('ringer_v8_1/2_loose', v8_half_loose),
#               ('ringer_v8_1/2_veryloose', v8_half_vloose),
             ]


# ## Setup Chains:

# In[ ]:


# create my chains

#e_24
my_chain_e24_v8_tight = Chain( "HLT_e24_lhtight_nod0_ivarloose_v8_tight", 
                              L1Seed = 'L1_EM22VHI', 
                              l2calo_column = 'ringer_v8_tight' )

my_chain_e24_v8_medium = Chain( "HLT_e24_lhtight_nod0_ivarloose_v8_medium", 
                               L1Seed = 'L1_EM22VHI', 
                               l2calo_column = 'ringer_v8_medium' )

my_chain_e24_v8_loose = Chain( "HLT_e24_lhtight_nod0_ivarloose_v8_loose", 
                              L1Seed = 'L1_EM22VHI', 
                              l2calo_column = 'ringer_v8_loose' )

my_chain_e24_v8_vloose = Chain( "HLT_e24_lhtight_nod0_ivarloose_v8_vloose", 
                               L1Seed = 'L1_EM22VHI', 
                               l2calo_column = 'ringer_v8_veryloose' )

my_chain_e24_v8_no_ringer = Chain( "e24_lhtight_nod0_noringer_ivarloose", 
                                  L1Seed = 'L1_EM22VHI' )
#e_26
my_chain_e26_v8_tight = Chain( "HLT_e26_lhtight_nod0_ivarloose_v8_tight", 
                              L1Seed = 'L1_EM22VHI', 
                              l2calo_column = 'ringer_v8_tight' )

my_chain_e26_v8_medium = Chain( "HLT_e26_lhtight_nod0_ivarloose_v8_medium", 
                               L1Seed = 'L1_EM22VHI', 
                               l2calo_column = 'ringer_v8_medium' )

my_chain_e26_v8_loose = Chain( "HLT_e26_lhtight_nod0_ivarloose_v8_loose", 
                              L1Seed = 'L1_EM22VHI', 
                              l2calo_column = 'ringer_v8_loose' )

my_chain_e26_v8_vloose = Chain( "HLT_e26_lhtight_nod0_ivarloose_v8_vloose", 
                               L1Seed = 'L1_EM22VHI', 
                               l2calo_column = 'ringer_v8_veryloose' )

my_chain_e26_v8_no_ringer = Chain( "e26_lhtight_nod0_noringer_ivarloose", 
                                  L1Seed = 'L1_EM22VHI' )

#e_60
my_chain_e60_v8_tight = Chain( "HLT_e60_lhmedium_nod0_v8_tight", 
                              L1Seed = 'L1_EM22VHI', 
                              l2calo_column = 'ringer_v8_tight' )


# In[ ]:


chains = [
            ("e24_lhtight_nod0_ivarloose_v8_tight", my_chain_e24_v8_tight),
            ("e24_lhtight_nod0_ivarloose_v8_medium", my_chain_e24_v8_medium),
            ("e24_lhtight_nod0_ivarloose_v8_loose", my_chain_e24_v8_loose),
            ("e24_lhtight_nod0_ivarloose_v8_vloose", my_chain_e24_v8_vloose),
            ("e24_lhtight_nod0_noringer_ivarloose", my_chain_e24_v8_no_ringer),
            ("e26_lhtight_nod0_ivarloose_v8_tight", my_chain_e26_v8_tight),
            ("e26_lhtight_nod0_ivarloose_v8_medium", my_chain_e26_v8_medium),
            ("e26_lhtight_nod0_ivarloose_v8_loose", my_chain_e26_v8_loose),
            ("e26_lhtight_nod0_ivarloose_v8_vloose", my_chain_e26_v8_vloose),
            ("e26_lhtight_nod0_noringer_ivarloose", my_chain_e26_v8_no_ringer),
            ("e60_lhmedium_nod0_v8_tight", my_chain_e60_v8_tight),
         ]


# ## Read all bins:

# In[ ]:


def read_all_files( path , decorators, chains = [], drop_rings=True, drop_other_columns = [] ):
    df_list = []
    for et_bin in range(5):    
        for eta_bin in range(0,1):
            print(path.format(ET=et_bin, ETA=eta_bin ))
            df_temp = load( path.format(ET=et_bin, ETA=eta_bin ) )
            
            # propagate ringer
            for column, decorator in decorators:
                # Apply v8 column
                decorator.apply( df_temp, column )
                
            # emulate chains
            for column, chain in chains:
                chain.apply(df_temp, column)
            
            if drop_rings:
                drop_ring_columns(df_temp)
            if drop_other_columns:
                df_temp.drop( drop_other_columns, axis=1, inplace=True )
                
            df_list.append(df_temp)
            
    return pd.concat(df_list)
                           
df = read_all_files(dpath, decorators=decorators, chains=chains, drop_other_columns=drop_columns, drop_rings=True)    


# In[ ]:


df = df.reset_index(drop=True)


# In[ ]:


list(df.columns)


# In[ ]:


np.sum(df['L1Calo_e24_lhtight_nod0_ivarloose_v8_tight'].astype(int)-df['L1Calo_e60_lhmedium_nod0_v8_tight'].astype(int))


# In[ ]:


# criar um vetor de eficiencia para cada um dos bins de uma variável 

variable = "el_et"
ref_var = 'target'
trigger_var = 'L2Calo_e24_lhtight_nod0_ivarloose'

comparison_columns = [variable, ref_var, trigger_var]

# criar um df de trabalho com as colunas de comparação
work_df = df[comparison_columns].copy(deep=True)

# usando uma definição básica para desenvolver
bins = [work_df[variable].min(),
        work_df[variable].quantile(q=0.25), 
        work_df[variable].quantile(q=0.50), 
        work_df[variable].quantile(q=0.75),
        work_df[variable].max()
       ]

# a comparação será feita utilizando como referencia a coluna ref_var
trigger_ref_on_eff = np.zeros([len(bins)-1])
trigger_all_eff = np.zeros([len(bins)-1])
for ibin, bin_value in enumerate(bins):
    if ibin == len(bins)-1:
        break
    binned_df = work_df.query('%1.2f < %s <= %1.2f'%(bins[ibin],variable, bins[ibin+1]))
    binned_ref_on = binned_df.query('%s ==1'%(ref_var))
    if binned_df.shape[0] != 0:
        trigger_ref_on_eff[ibin] = float(binned_ref_on[trigger_var].sum()/
                                         binned_ref_on.shape[0])
        trigger_all_eff[ibin] = float(binned_df[trigger_var].sum()/
                                      binned_df.shape[0])


# In[ ]:


def get_trg_eff_4_var(df, variable, trigger_var, bins=None, trgt_on_only=True):
    '''This function estimate the efficiency for a specific variable in a spec. trigger
    df: input dataframe
    variable: variable to be analyzed
    trigger_var: trigger response variable (1=accepted, 0=rejected)
    bins: None (default), number of bins (int), vector
    '''
    
    # filtrar a saida do offline da cadeia - trigger_var
    # filtrar target == 1 pois estamos falando de eff
    # colocar o PID
    # na variaveld e trigger, temos a informação do offline que deve ser utilizado
    offline_selection = trigger_var.split('_')[2]
    offline_selection = 'el_'+offline_selection
    
    # get variables
    comparison_columns = [variable, trigger_var, offline_selection, 'target']
    # creating a working df to manipulate everything
    work_df = df[comparison_columns].copy(deep=True)
    
    # applying selection
    work_df = work_df.query('target==1 & %s == 1'%(offline_selection))

    # choosing bins
    if bins is None:
        nbins = 10
        bins = np.zeros([nbins])
        for ibin in range(0,nbins):
            if ibin == 0:
                bins = [work_df[variable].min()]
            else:
                bins.append(work_df[variable].quantile(q=ibin/float(nbins)))
    elif type(bins) == int:
        nbins = bins
        bins = np.linspace(work_df[variable].min(),work_df[variable].max(),nbins)
        
    bins_center = np.zeros([len(bins)-1])
    
    trigger_eff = np.zeros([len(bins)-1])
    trigger_var_qtd = np.zeros([len(bins)-1])
    trigger_ref_qtd = np.zeros([len(bins)-1])
    for ibin, bin_value in enumerate(bins):
        if ibin == len(bins)-1:
            break
        bins_center[ibin] = bins[ibin]+ ((bins[ibin+1]-bins[ibin])/2)
        binned_df = work_df.query('%1.2f < %s <= %1.2f'%(bins[ibin],variable, bins[ibin+1]))
        if binned_df.shape[0] != 0:
            # all quantities variables
            trigger_var_qtd[ibin] = float(binned_df[trigger_var].sum())
            trigger_ref_qtd[ibin] = float(binned_df.shape[0])
            trigger_eff[ibin] = float(binned_df[trigger_var].sum()/
                                      binned_df.shape[0])
    return trigger_eff, trigger_var_qtd, trigger_ref_qtd, bins, bins_center


# In[ ]:


# plot um gráfico com duas figuras.
# a primeira é o número de eventos por bin
# a segunda é a eficiência por bin

from matplotlib import gridspec

# criar uma figura
figsize = (13,12)
nrows = 2
ncols = 1
fig = plt.figure(figsize=figsize)
    
# create grid for different subplots
spec = gridspec.GridSpec(ncols=1, nrows=2,
width_ratios=[1],
height_ratios=[2.5, 1],
wspace=0.0, hspace=0.1,)

ax  = fig.add_subplot(spec[0]) # big figure
ax1 = fig.add_subplot(spec[1]) # secondary axis
ax1.sharex(ax)

ax.tick_params(axis="x", colors="None")

data = [df]
labels = ['sgn',]
colors = ['blue']

variable = 'el_et'
chain = "L2Calo_e24_lhtight_nod0_ivarloose"
trigger_var = chain+'_v8_tight'

etbins=[0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 
        14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 
        26.0, 28.0, 30.0, 32.0, 34.0, 36.0, 
        38.0, 40.0, 42.0, 44.0, 46.0, 48.0, 
        50.0, 55.0, 60.0, 65.0, 70.0, 100.0]

bins = np.array(etbins)*1000
#bins = 10

for idb, idata in enumerate(data):
    print('Processing data of dataset: %s'%(labels[idb]))
    [eff, var_qtd, 
     ref_qtd, bins, 
     bins_center] = get_trg_eff_4_var(df=df,variable=variable,
                                      trigger_var=trigger_var,
                                      bins = bins
                                     )
    
    n, m_bins, patches = ax.hist(idata[variable], bins, density=False,
                                 color=colors[idb], alpha=1.0, 
                                 label=labels[idb], histtype='step', 
                                 lw=1.5)
    ax1.plot(bins_center,eff,'o',color=colors[idb], alpha=0.75, label=labels[idb])

# plot ajusts
ax.grid()
ax.set_yscale('log')
ax.set_ylabel('Counts')
ax.legend(bbox_to_anchor=(0,0.92,1,0.2), mode='expand', loc="upper left", ncol=3)

ax1.grid()
ax1.set_ylim([0.01, 1.5])
ax1.set_xlabel(variable)
ax1.set_ylabel('Trigger Eff')
ax1.set_yscale('log')

hep.atlas.text(text='Internal', loc=1, ax=ax)
ax.text(0.05, 0.85, 'data17\n\n$\sqrt{s}$= $13\ TeV$\n\nPID: Egamma1\n\nChain: %s'%(chain), 
        horizontalalignment='left', 
        verticalalignment='top',
        transform=ax.transAxes,
        fontsize=14,);
plt.savefig('trigger_eff_chain_%s_var_%s.png'%(trigger_var,variable))


# In[ ]:


# plotar a eficiencia em diferentes pontos da cadeia


# In[ ]:


# plotar a eficiencia para diferentes versions do trigger nos mesmos pontos

