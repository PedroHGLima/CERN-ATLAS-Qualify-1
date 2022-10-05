export MY_CURRENT_PATH=$PWD

source build_image.sh
source build_container.sh

docker start qualify_cern
docker exec -it qualify_cern bash

cd $MY_CURRENT_PATH
