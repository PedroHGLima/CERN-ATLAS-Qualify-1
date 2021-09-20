import numpy as np
from prometheus import EventATLAS
from prometheus.enumerations import Dataframe as DataframeEnum
from Gaugi.messenger import LoggingLevel, Logger
from Gaugi import ToolSvc, ToolMgr
import argparse
mainLogger = Logger.getModuleLogger("job")
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()


parser.add_argument('-i','--inputFiles', action='store', 
    dest='inputFiles', required = True, nargs='+',
    help = "The input files that will be used to generate the plots")

parser.add_argument('-o','--outputFile', action='store', 
    dest='outputFile', required = False, default = None,
    help = "The output store name.")

parser.add_argument('-n','--nov', action='store', 
    dest='nov', required = False, default = -1, type=int,
    help = "Number of events.")

parser.add_argument('--fake', action='store_true', 
    dest='fakes', required = False, 
    help = "Set the pidname and treePath for background.")

import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()


if args.fakes:
  pidname       = '!el_lhvloose'
  #m_treePath    = '*/HLT/Physval/Egamma/fakes'
  m_treePath    = '*/HLT/Egamma/Egamma/fakes'
else:
  m_treePath = '*/HLT/Egamma/Egamma/probes'

acc = EventATLAS( "EventATLASLoop",
                  inputFiles = args.inputFiles, 
                  treePath= m_treePath,
                  dataframe = DataframeEnum.Electron_v1, 
                  outputFile = args.outputFile,
                  level = LoggingLevel.INFO
                  )


# EventSelection Configuration
from EventSelectionTool import EventSelection, SelectionType, EtCutType
evt = EventSelection('EventSelection')
evt.setCutValue( SelectionType.SelectionOnlineWithRings )

evt.setCutValue( EtCutType.L2CaloAbove , 15)

ToolSvc += evt


from TrigEgammaEmulationTool import Chain, Group, TDT
# Trigger list 
 
triggerList = [
    # e24_lhtight_nod0
    Group(Chain('EMU_e24_lhtight_nod0_ringer_v8_ivarloose',
                'L1_EM3', 'HLT_e24_lhtight_nod0_ringer_v8_ivarloose'), 'el_lhtight', 24),
    Group(Chain('EMU_e24_lhtight_nod0_ringer_v8.1_ivarloose',
                'L1_EM3', 'HLT_e24_lhtight_nod0_ringer_v8.1_ivarloose'), 'el_lhtight', 24),
    Group(Chain('EMU_e24_lhtight_nod0_ringer_v9_ivarloose',
                'L1_EM3', 'HLT_e24_lhtight_nod0_ringer_v9_ivarloose'), 'el_lhtight', 24),
    Group(Chain('EMU_e24_lhtight_nod0_noringer_ivarloose',
                'L1_EM3', 'HLT_e24_lhtight_nod0_noringer_ivarloose'), 'el_lhtight', 24),
    # e26_lhtight_nod0
    Group(Chain('EMU_e26_lhtight_nod0_ringer_v8_ivarloose',
                'L1_EM3', 'HLT_e26_lhtight_nod0_ringer_v8_ivarloose'), 'el_lhtight', 26),
    Group(Chain('EMU_e26_lhtight_nod0_ringer_v8.1_ivarloose',
                'L1_EM3', 'HLT_e26_lhtight_nod0_ringer_v8.1_ivarloose'), 'el_lhtight', 26),                
    Group(Chain('EMU_e26_lhtight_nod0_ringer_v9_ivarloose',
                'L1_EM3', 'HLT_e26_lhtight_nod0_ringer_v9_ivarloose'), 'el_lhtight', 26),
    Group(Chain('EMU_e26_lhtight_nod0_noringer_ivarloose',
                'L1_EM3', 'HLT_e26_lhtight_nod0_noringer_ivarloose'), 'el_lhtight', 26),
    # e60_lhmedium_nod0
    Group(Chain('EMU_e60_lhmedium_nod0_ringer_v8',
                'L1_EM3', 'HLT_e60_lhmedium_nod0_ringer_v8'), 'el_lhmedium', 60),
    Group(Chain('EMU_e60_lhmedium_nod0_ringer_v8.1',
                'L1_EM3', 'HLT_e60_lhmedium_nod0_ringer_v8.1'), 'el_lhmedium', 60),
    Group(Chain('EMU_e60_lhmedium_nod0_ringer_v9',
                'L1_EM3', 'HLT_e60_lhmedium_nod0_ringer_v9'), 'el_lhmedium', 60),
    Group(Chain('EMU_e60_lhmedium_nod0_noringer',
                'L1_EM3', 'HLT_e60_lhmedium_nod0_noringer'), 'el_lhmedium', 60),  
    # e140_lhloose_nod0
    Group(Chain('EMU_e140_lhloose_nod0_ringer_v8',
                'L1_EM3', 'HLT_e140_lhloose_nod0_ringer_v8'), 'el_lhloose', 140),
    Group(Chain('EMU_e140_lhloose_nod0_ringer_v8.1',
                'L1_EM3', 'HLT_e140_lhloose_nod0_ringer_v8.1'), 'el_lhloose', 140),
    Group(Chain('EMU_e140_lhloose_nod0_ringer_v9',
                'L1_EM3', 'HLT_e140_lhloose_nod0_ringer_v9'), 'el_lhloose', 140),
    Group(Chain('EMU_e140_lhloose_nod0_noringer',
                'L1_EM3', 'HLT_e140_lhloose_nod0_noringer'), 'el_lhloose', 140),
]

from EfficiencyTools import EfficiencyTool
alg = EfficiencyTool( "Efficiency", dojpsiee=False , eta_bins = np.arange(-2.0,2.0,0.5))

for group in triggerList:
  alg.addGroup( group )

ToolSvc += alg

acc.run(args.nov)