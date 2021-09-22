prun_jobs.py -c "python3 job_zee_efficiency.py" -mt 40 -i /home/natmourajr/Workspace/CERN/Qualify/data/PhysVal_v2/EGAM1/after_ts1/*

# run one file for boosted electrons
# -i /home/natmourajr/Workspace/CERN/Qualify/data/Zee_boosted/user.jodafons.mc16_13TeV.302236.MadGraphPythia8EvtGen_A14NNPDF23LO_HVT_Agv1_VcWZ_llqq_m3000.merge.AOD.ID6.t0_GLOBAL/user.jodafons.26370333.GLOBAL._000001.root
# run all files for boosted electrons
# -i /home/natmourajr/Workspace/CERN/Qualify/data/Zee_boosted/*
# to run on Zee events
# -i /home/natmourajr/Workspace/CERN/Qualify/data/PhysVal_v2/EGAM1/after_ts1/*


mkdir egam1_test
prun_merge.py -i output_* -o egam1_test.root -nm 35 -mt 8
mv egam1_test.root egam1_test
rm -rf output_*

