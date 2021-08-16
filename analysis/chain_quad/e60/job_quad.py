
import argparse
from prometheus import EventATLAS
from prometheus.enumerations import Dataframe as DataframeEnum
from Gaugi.messenger import LoggingLevel, Logger
from Gaugi import ToolSvc, ToolMgr


jr_verbose = False

if jr_verbose: print(1)
mainLogger = Logger.getModuleLogger("job")
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

if jr_verbose: print(2)

parser.add_argument('-i','--inputFiles', action='store', 
    dest='inputFiles', required = True, nargs='+',
    help = "The input files that will be used to generate the plots")

parser.add_argument('-o','--outputFile', action='store', 
    dest='outputFile', required = False, default = None,
    help = "The output store name.")

parser.add_argument('-n','--nov', action='store', 
    dest='nov', required = False, default = -1, type=int,
    help = "Number of events.")

parser.add_argument('--Zee', action='store_true', 
    dest='doZee', required = False, 
    help = "Do Zee collection.")

parser.add_argument('--egam7', action='store_true', 
    dest='doEgam7', required = False, 
    help = "The colelcted sample came from EGAM7 skemma.")


if jr_verbose: print(3)
import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()

acc = EventATLAS( "EventATLASLoop",
                  inputFiles = args.inputFiles, 
                  treePath= '*/HLT/Physval/Egamma/probes',
                  dataframe = DataframeEnum.Electron_v1, 
                  outputFile = args.outputFile,
                  level = LoggingLevel.INFO
                  )

if jr_verbose: print(4)

from EventSelectionTool import EventSelection, SelectionType, EtCutType
evt = EventSelection('EventSelection')
evt.setCutValue( SelectionType.SelectionOnlineWithRings )

# Do not change this!
if args.doEgam7:
  #pidname = '!VeryLooseLLH_DataDriven_Rel21_Run2_2018'
  pidname = '!el_lhvloose'
else:
  #pidname = 'MediumLLH_DataDriven_Rel21_Run2_2018'
  pidname = 'el_lhtight'
  #pidname = 'el_lhmedium'
  #pidname  = 'el_lhvloose'

evt.setCutValue( SelectionType.SelectionPID, pidname ) 
#evt.setCutValue( EtCutType.L2CaloAbove, 3.)
evt.setCutValue( EtCutType.L2CaloAbove, 15.)
#evt.setCutValue( EtCutType.OfflineAbove, 2.)

if jr_verbose: print(5)

ToolSvc += evt

if jr_verbose: print(5.1)

from TrigEgammaEmulationTool import Chain

triggerList = [
                Chain( "EMU_e60_lhmedium_nod0_ringer_v8" ,  "L1_EM3", "HLT_e60_lhmedium_nod0_ringer_v8"  ),
                Chain( "EMU_e60_lhmedium_nod0_ringer_v11",  "L1_EM3", "HLT_e60_lhmedium_nod0_ringer_v11" ),
              ]

if jr_verbose: print(6)

# Add all chains into the emulator
emulator = ToolSvc.retrieve( "Emulator" )
for chain in triggerList:
  print(chain.name())
  if not emulator.isValid( chain.name() ):
    emulator+=chain

if jr_verbose: print(7)

from QuadrantTools import QuadrantTool
q_alg = QuadrantTool("Quadrant")
q_alg.add_quadrant( "HLT_e60_lhmedium_nod0_ringer_v8"  , "EMU_e60_lhmedium_nod0_ringer_v8", # Ringer v8
                    'EMU_e60_lhmedium_nod0_ringer_v11' , "EMU_e60_lhmedium_nod0_ringer_v11" # Ringer v11
                  ) 

if jr_verbose: print(8)

# vloose chain
#q_alg.add_quadrant( 'HLT_e5_lhvloose_nod0_noringer'  , "EMU_e5_lhvloose_nod0_noringer", # T2Calo
#                    'HLT_e5_lhvloose_nod0_ringer_v1' , "EMU_e5_lhvloose_nod0_ringer_v1" # Ringer v1
#                  ) 

if jr_verbose: print(9)

etlist = [15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,50000.0]
etalist= [ 0.0, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47 ]
#etlist = [3.0, 7.0, 10.0, 15.0]
#etalist= [ 0.0, 0.8, 1.37, 1.54, 2.37, 2.47]
q_alg.setEtBinningValues(etlist)
q_alg.setEtaBinningValues(etalist)
ToolSvc += q_alg

#from ImpactTools import ImpactTool
#i_alg = ImpactTool("Impact", dataframe = DataframeEnum.Electron_v1)
#i_alg.add_selection(  'HLT_e5_lhtight_nod0_noringer'  , "EMU_e5_lhtight_nod0_noringer", # T2Calo
#                      'HLT_e5_lhtight_nod0_ringer_v1' , "EMU_e5_lhtight_nod0_ringer_v1", # Ringer v1
#                    ) 

#etlist = [3.0, 7.0, 10.0, 15.0]
#etalist= [ 0.0, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47 ]
#etalist= [ 0.0, 0.8, 1.37, 1.54, 2.37, 2.47]
#i_alg.setEtBinningValues(etlist)
#i_alg.setEtaBinningValues(etalist)
#ToolSvc += i_alg

if jr_verbose: print(10)

acc.run(args.nov)

if jr_verbose: print(11)
#/home/natmourajr/Workspace/CERN/Qualify/data/PhysVal_v2/EGAM1/after_ts1/user.jodafons.data17_13TeV.00339205.physics_Main.deriv.DAOD_EGAM1.f887_m1897_p3336.Physval.GRL_v97.r7000_GLOBAL
#user.jodafons.13861574.GLOBAL._000005.root