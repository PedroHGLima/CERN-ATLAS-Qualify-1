import argparse
from prometheus import EventATLAS
from prometheus.enumerations import Dataframe as DataframeEnum
from Gaugi.messenger import LoggingLevel, Logger
from Gaugi import ToolSvc, ToolMgr


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

parser.add_argument('--Zee', action='store_true', 
    dest='doZee', required = False, 
    help = "Do Zee collection.")

parser.add_argument('--egam7', action='store_true', 
    dest='doEgam7', required = False, 
    help = "The colelcted sample came from EGAM7 skemma.")


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

from EventSelectionTool import EventSelection, SelectionType, EtCutType
evt = EventSelection('EventSelection')
evt.setCutValue( SelectionType.SelectionOnlineWithRings )

# since we are trying to validate a tuning 
# we need to reproduce the same dataset that was used in the training process
if args.doEgam7:
  pidname = '!el_lhvloose'
else:
  pidname = 'el_lhmedium'

# apply selections
evt.setCutValue( SelectionType.SelectionPID, pidname ) 
evt.setCutValue( EtCutType.L2CaloAbove, 15.)
#evt.setCutValue( EtCutType.L2CaloAbove, 4.)
#evt.setCutValue( EtCutType.L2CaloBelow, 15.)
#evt.setCutValue( EtCutType.OfflineAbove, 2.)

# add selection tool to the processing pipelin
ToolSvc += evt

# initialize the emulator
from TrigEgammaEmulationTool import installElectronL2CaloRingerSelector_v8, installElectronL2CaloRingerSelector_v11

# Add all chains into the emulator
emulator = ToolSvc.retrieve( "Emulator" )
# install selectors
# search on prometheus/trigger/install.py
installElectronL2CaloRingerSelector_v8()
installElectronL2CaloRingerSelector_v11()


# initialize the quadrant tool
from QuadrantTools import QuadrantTool
q_alg = QuadrantTool("Quadrant")

q_alg.add_quadrant( 
                # tight
                'ringer_v8_tight', 'T0HLTElectronRingerTight_v8', # Ringer v8
                'ringer_v11_tight', 'T0HLTElectronRingerTight_v11'  # Ringer v11
                )
q_alg.add_quadrant( 
                # medium
                'ringer_v8_medium', 'T0HLTElectronRingerMedium_v8', # Ringer v8
                'ringer_v11_medium', 'T0HLTElectronRingerMedium_v11'  # Ringer v11
                )
q_alg.add_quadrant( 
                # loose
                'ringer_v8_loose', 'T0HLTElectronRingerLoose_v8', # Ringer v8
                'ringer_v11_loose', 'T0HLTElectronRingerLoose_v11'  # Ringer v11
                )
q_alg.add_quadrant( 
                # very loose
                'ringer_v8_vloose', 'T0HLTElectronRingerVeryLoose_v8', # Ringer v8
                'ringer_v11_vloose', 'T0HLTElectronRingerVeryLoose_v11'  # Ringer v11
                )
# first let's run using the same binning thar was used in the training
#etlist = [3.0, 7.0, 10.0, 15.0]
etlist = [15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,50000.0]

#etalist= [0.0, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47]
etalist= [ 0.0, 0.8, 1.37, 1.54, 2.37, 2.47]
q_alg.setEtBinningValues(etlist)
q_alg.setEtaBinningValues(etalist)

# add to the pile
ToolSvc += q_alg

acc.run(args.nov)