import argparse
from prometheus import EventATLAS
from prometheus.enumerations import Dataframe as DataframeEnum
from Gaugi.messenger import LoggingLevel, Logger
from Gaugi import ToolSvc, ToolMgr

mainLogger = Logger.getModuleLogger("job")
parser = argparse.ArgumentParser(description = '', add_help = False)
parser = argparse.ArgumentParser()

parser.add_argument('-i','--inputFile', action='store', 
    dest='inputFile', required = True,
    help = "The input files that will be used to generate the plots")


import sys,os
if len(sys.argv)==1:
  parser.print_help()
  sys.exit(1)

args = parser.parse_args()


from QuadrantTools import QuadrantTool
q_alg = QuadrantTool("Quadrant")

from Gaugi  import restoreStoreGate
sg =  restoreStoreGate( args.inputFile )
q_alg.setStoreGateSvc(sg)

q_alg.add_quadrant( 
                # tight
                'ringer_v8_tight', 'T0HLTElectronRingerTight_v8', # Ringer v8
                'ringer_v9_tight', 'T0HLTElectronRingerTight_v9'  # Ringer v9
                )
q_alg.add_quadrant( 
                # medium
                'ringer_v8_medium', 'T0HLTElectronRingerMedium_v8', # Ringer v8
                'ringer_v9_medium', 'T0HLTElectronRingerMedium_v9'  # Ringer v9
                )
q_alg.add_quadrant( 
                # loose
                'ringer_v8_loose', 'T0HLTElectronRingerLoose_v8', # Ringer v8
                'ringer_v9_loose', 'T0HLTElectronRingerLoose_v9'  # Ringer v9
                )
q_alg.add_quadrant( 
                # very loose
                'ringer_v8_vloose', 'T0HLTElectronRingerVeryLoose_v8', # Ringer v8
                'ringer_v9_vloose', 'T0HLTElectronRingerVeryLoose_v9'  # Ringer v9
                )

#etlist = [3.0, 7.0, 10.0, 15.0]
etlist = [15.0,20.0,25.0,30.0,35.0,40.0,45.0,50.0,50000.0]
#etalist= [ 0.0, 0.6, 0.8, 1.15, 1.37, 1.52, 1.81, 2.01, 2.37, 2.47 ]
etalist= [ 0.0, 0.8, 1.37, 1.54, 2.37, 2.47]
q_alg.setEtBinningValues(etlist)
q_alg.setEtaBinningValues(etalist)
ToolSvc += q_alg


outputs = [
            'quadrant_data17_egam2_lhtight_ringer_v8_and_ringer_v9',
            'quadrant_data17_egam2_lhmedium_ringer_v8_and_ringer_v9',
            'quadrant_data17_egam2_lhloose_ringer_v8_and_ringer_v9',
            'quadrant_data17_egam2_lhvloose_ringer_v8_and_ringer_v9',
            ]

legends = ['Both Approved', 'Ringer V8 Only Approved', 'Ringer V9 Only Approved', 'Both Rejected']

names   = [
            'Quadrant Analysis lhtight Ringer V8 vs Ringer V9 (data17-EGAM1)',
            'Quadrant Analysis lhmedium Ringer V8 vs Ringer V9 (data17-EGAM1)',
            'Quadrant Analysis lhloose Ringer V8 vs Ringer V9 (data17-EGAM1)',
            'Quadrant Analysis lhvloose Ringer V8 vs Ringer V9 (data17-EGAM1)',
            ]

q_alg.plot(outputs, outputs, names,legends=legends, doPDF=True)