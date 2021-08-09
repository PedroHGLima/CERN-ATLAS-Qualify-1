export MY_CURRENT_PATH=$PWD
cd /home/natmourajr/Workspace/UFRJ/CERN/CERN-ATLAS-Qualify/docker_images/ringer-jr
docker build --tag=qualify_image:v1.0 .
echo 'rodou...'
cd $MY_CURRENT_PATH
