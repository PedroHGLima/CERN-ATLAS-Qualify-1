prun_jobs.py -c "python job_quad.py --Zee" \
                    -i /home/natmourajr/Workspace/CERN/Qualify/data/PhysVal_v2/EGAM1/after_ts1/user.jodafons.data17_13TeV.00339205.physics_Main.deriv.DAOD_EGAM1.f887_m1897_p3336.Physval.GRL_v97.r7000_GLOBAL/user.jodafons.13861574.GLOBAL._0000* -mt 6

mkdir egam1_test
prun_merge.py -i output_* -o egam1_test.root -nm 35 -mt 8
mv egam1_test.root egam1_test
rm -rf output_*