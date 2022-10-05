
github_username=$1


echo "======================================================================================"
echo "setup root..."
# Set ROOT by hand
export ROOTSYS="/opt/root/buildthis"
export PATH="$ROOTSYS/bin:$PATH"
export LD_LIBRARY_PATH="$ROOTSYS/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="/opt/root/buildthis/lib:$PYTHONPATH"

echo "======================================================================================"
echo "setup gaugi..."
current=$PWD
cd $(mktemp -d)
mkdir git_repos
cd git_repos
temp=$PWD

cd $temp && git clone https://github.com/$github_username/gaugi.git
cd $temp/gaugi && source scripts/setup.sh
echo "======================================================================================"
echo "setup prometheus..."
cd $temp && git clone https://github.com/$github_username/prometheus.git
cd $temp/prometheus && mkdir build && cd build && cmake .. && make && cd .. && source setup.sh

echo "======================================================================================"
echo "setup kolmov..."
cd $temp && git clone https://github.com/$github_username/kolmov.git
cd $temp/kolmov && source scripts/setup.sh

echo "======================================================================================"
echo "setup saphyra..."
cd $temp && git clone https://github.com/$github_username/saphyra.git
cd $temp/saphyra && source scripts/setup.sh


echo "======================================================================================"
echo "setup orchestra..."
cd $temp && git clone https://github.com/$github_username/orchestra.git
cd $temp/orchestra && source setup.sh


echo "======================================================================================"
echo "setup tunings..."
cd $temp && git clone https://github.com/$github_username/ringer_tunings.git
cd $temp/ringer_tunings && source setup.sh

echo "======================================================================================"
echo "all packages can be found at $temp."
echo "======================================================================================"
cd $current