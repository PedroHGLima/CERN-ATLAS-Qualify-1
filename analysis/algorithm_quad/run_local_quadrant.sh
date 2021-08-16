prun_jobs.py -c "python job_local_quadrant.py --Zee" -i /home/natmourajr/Workspace/CERN/Qualify/data/PhysVal_v2/EGAM1/after_ts1/* -mt 40


# to run one file only
# -i /home/natmourajr/Workspace/CERN/Qualify/data/PhysVal_v2/EGAM1/after_ts1/user.jodafons.data17_13TeV.00339205.physics_Main.deriv.DAOD_EGAM1.f887_m1897_p3336.Physval.GRL_v97.r7000_GLOBAL/user.jodafons.13861574.GLOBAL._000005.root
# to run all files
# -i /home/natmourajr/Workspace/CERN/Qualify/data/PhysVal_v2/EGAM1/after_ts1/*

mkdir local_quadrant_egam1
prun_merge.py -i output_* -o local_quadrant_egam1.root -nm 35 -mt 8
mv local_quadrant_egam1.root local_quadrant_egam1
rm -rf output_*