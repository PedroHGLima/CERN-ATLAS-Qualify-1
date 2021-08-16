from Gaugi.messenger import LoggingLevel, Logger
from Gaugi.storage import restoreStoreGate
#from EfficiencyTools import PlotProfiles, GetProfile
from EfficiencyTools import GetProfile
from old_plot_profiles import PlotProfiles

import ROOT

ROOT.gROOT.SetBatch(True)

mainLogger = Logger.getModuleLogger("job")
mainLogger.level = LoggingLevel.INFO

theseColors = [ROOT.kBlack, ROOT.kGray+2, ROOT.kBlue-2, ROOT.kBlue-4]

def plot_table( sg, logger, trigger, basepath ):
  triggerLevels = ['L1Calo','L2Calo','L2','EFCalo','HLT']
  logger.info( '{:-^78}'.format((' %s ')%(trigger)) ) 
  
  for trigLevel in triggerLevels:
    dirname = basepath+'/'+trigger+'/Efficiency/'+trigLevel
    total  = sg.histogram( dirname+'/eta' ).GetEntries()
    passed = sg.histogram( dirname+'/match_eta' ).GetEntries()
    eff = passed/float(total) * 100. if total>0 else 0
    eff=('%1.2f')%(eff); passed=('%d')%(passed); total=('%d')%(total)
    stroutput = '| {0:<30} | {1:<5} ({2:<5}, {3:<5}) |'.format(trigLevel,eff,passed,total)
    logger.info(stroutput)
  logger.info( '{:-^78}'.format((' %s ')%('-')))


def get( sg, path, histname, resize=None ):
  return GetProfile( sg.histogram( path + '/match_'+histname), sg.histogram( path + '/' + histname ), resize=resize )



inputFile = 'egam1_test/egam1_test.root'
basepath = 'Event/EfficiencyTool'

sg =  restoreStoreGate( inputFile )

chain_dict_config = {
  # e26_lhtight_nod0
  'HLT_e26_lhtight_nod0_ivarloose' : {
                            'triggers' : ["EMU_e26_lhtight_nod0_ringer_v8_ivarloose", 
                                          "EMU_e26_lhtight_nod0_ringer_v11_ivarloose",
                                          "EMU_e26_lhtight_nod0_noringer_ivarloose"],
                            'plotname' : 'efficiency_v1_boosted_%s_%s_e26_lhtight_nod0_ivarloose_eff',
                            },
  # e60_lhmedium_nod0
  'HLT_e60_lhmedium_nod0' : {
                            'triggers' : ["EMU_e60_lhmedium_nod0_ringer_v8", 
                                          "EMU_e60_lhmedium_nod0_ringer_v11",
                                          "EMU_e60_lhmedium_nod0_noringer"],
                            'plotname' : 'efficiency_v1_boosted_%s_%s_e60_lhmedium_nod0_eff',
                            },
  # e140_lhloose_nod0
  'HLT_e140_lhloose_nod0' : {
                            'triggers' : ["EMU_e140_lhloose_nod0_ringer_v8", 
                                          "EMU_e140_lhloose_nod0_ringer_v11",
                                          "EMU_e140_lhloose_nod0_noringer"],
                            'plotname' : 'efficiency_v1_boosted_%s_%s_e140_lhloose_nod0_eff',
                            },
}


for ichain in chain_dict_config.keys():
  triggers = chain_dict_config[ichain]['triggers']
  print('Chain: %s' %(ichain))
  for istep in ['L2Calo', 'HLT']:
    plotname = chain_dict_config[ichain]['plotname'] %(inputFile.split('/')[0], istep)
          
    eff_et  = [ get(sg, basepath+'/'+trigger+'/Efficiency/%s'%(istep), 'et') for trigger in triggers ]
    eff_eta = [ get(sg, basepath+'/'+trigger+'/Efficiency/%s'%(istep), 'eta') for trigger in triggers ]
    eff_phi = [ get(sg, basepath+'/'+trigger+'/Efficiency/%s'%(istep), 'phi') for trigger in triggers ]
    eff_mu  = [ get(sg, basepath+'/'+trigger+'/Efficiency/%s'%(istep), 'mu', [8,20,60]) for trigger in triggers ]
    eff_deltaR  = [ get(sg, basepath+'/'+trigger+'/Efficiency/%s'%(istep), 'deltaR') for trigger in triggers ]
    
    legends = ['ringer v8', 'ringer v11', 'no ringer']#, 'ringer old v1', 'ringer old v1 2']


    for trigger in triggers:
      plot_table( sg, mainLogger, trigger, basepath )

      PlotProfiles( eff_et, legends=legends,runLabel='mc16 13TeV', outname='%s_et.pdf' %(plotname), theseColors=theseColors,
                    extraText1=ichain,doRatioCanvas=False, legendX1=.65, xlabel='E_{T}', rlabel='Trigger/Ref.',ylabel='Trigger Efficiency')

      PlotProfiles( eff_eta, legends=legends,runLabel='mc16 13TeV', outname='%s_eta.pdf' %(plotname),theseColors=theseColors,
                    extraText1=ichain, doRatioCanvas=False, legendX1=.65, xlabel='#eta', rlabel='Trigger/Ref.',ylabel='Trigger Efficiency')

      PlotProfiles( eff_phi, legends=legends,runLabel='mc16 13TeV', outname='%s_phi.pdf' %(plotname),theseColors=theseColors,
                    extraText1=ichain, doRatioCanvas=False, legendX1=.65, xlabel='#phi', rlabel='Trigger/Ref.',ylabel='Trigger Efficiency')

      PlotProfiles( eff_mu, legends=legends,runLabel='mc16 13TeV', outname='%s_mu.pdf' %(plotname),theseColors=theseColors,
                    extraText1=ichain, doRatioCanvas=False, legendX1=.65, xlabel='<#mu>', rlabel='Trigger/Ref.',ylabel='Trigger Efficiency')

      PlotProfiles( eff_deltaR, legends=legends,runLabel='mc16 13TeV', outname='%s_deltaR.pdf' %(plotname),theseColors=theseColors,
                    extraText1=ichain, doRatioCanvas=False, legendX1=.65, xlabel='#Delta R', rlabel='Trigger/Ref.',ylabel='Trigger Efficiency')

