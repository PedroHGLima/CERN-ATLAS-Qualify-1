#!/bin/bash
sudo apt-get install dpkg-dev cmake g++ gcc binutils libx11-dev libxpm-dev libxft-dev libxext-dev python libssl-dev
sudo apt-get install gfortran libpcre3-dev xlibmesa-glu-dev libglew1.5-dev libftgl-dev libmysqlclient-dev libfftw3-dev libcfitsio-dev graphviz-dev libavahi-compat-libdnssd-dev libldap2-dev python-dev libxml2-dev libkrb5-dev libgsl0-dev 
sudo apt install cmake
sudo add-apt-repository ppa:rock-core/qt4
sudo apt update
sudo apt-get install libqt4-dev
sudo apt-get install git
git clone --branch v6-22-00-patches https://github.com/root-project/root.git root_src
mkdir root_build && cd root_build
#mkdir /usr/local/lib/root/
cmake -DCMAKE_INSTALL_PREFIX=/usr/local/lib/ -DPYTHON_EXECUTABLE=/usr/bin/python3 -DPYHTON_INCDIR=/usr/local/lib/include/ -DPYTHON_LIBDIR=/usr/lib ../root_src
sudo cmake --build . --target install
source /usr/local/lib/bin/thisroot.sh
echo 'source /usr/local/lib/bin/thisroot.sh' >> ~/.bashrc 


