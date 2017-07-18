#!/bin/bash

set -e # stop this script on command not returning 0

SYSTEM=`uname -m`
OLDDIR=`pwd`

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

mkdir PACKAGES


# # download and build boost
# wget -O PACKAGES/boost_1_56_0.tar.gz http://downloads.sourceforge.net/project/boost/boost/1.56.0/boost_1_56_0.tar.gz?use_mirror=switch
# cd $DIR/PACKAGES
# tar xfvz boost_1_56_0.tar.gz
# cd boost_1_56_0
# ./bootstrap.sh --prefix=${DIR}/PACKAGES/boost_1_56_0/build
# sed -i -e 's/libraries =  ;/libraries = --with-python ;/g' ${DIR}/PACKAGES/boost_1_56_0/project-config.jam
# ./b2 install

# download ADOL-C
cd $DIR/PACKAGES
wget http://www.coin-or.org/download/source/ADOL-C/ADOL-C-2.6.0.tgz
tar xfvz ADOL-C-2.6.0.tgz
mv ADOL-C-2.6.0 ADOL-C
# git clone https://git.gitorious.org/adol-c/adol-c.git PACKAGES/ADOL-C
# git clone https://github.com/b45ch1/adol-c.git PACKAGES/ADOL-C
cd $DIR/PACKAGES/ADOL-C
# git checkout e686fc236baf2c109f9c7f43da9bbd88b558f74d
# git checkout 4f72634
cd $DIR

# download ColPack
#wget -P PACKAGES http://cscapes.cs.purdue.edu/download/ColPack/ColPack-1.0.9.tar.gz
wget -P PACKAGES https://github.com/CSCsw/ColPack/archive/v1.0.10.zip

# build ColPack
cd $DIR/PACKAGES
#tar xfvz ColPack-1.0.9.tar.gz
#cd ColPack-1.0.9
tar xfvz ColPack-1.0.10.tar.gz
cd ColPack-1.0.10
if [ SYSTEM = "x86_64" ]; then
autoreconf -vif
./configure --prefix=`pwd`/../ADOL-C/ThirdParty/ColPack/ --libdir='${prefix}/lib64'
else
autoreconf -vif
./configure --prefix=`pwd`/../ADOL-C/ThirdParty/ColPack/ --libdir='${prefix}/lib'
fi
#cd $DIR/PACKAGES/ColPack-1.0.9
cd $DIR/PACKAGES/ColPack-1.0.10
make
make install

# build ADOL-C
cd $DIR/PACKAGES/ADOL-C
./update_versions.sh
./configure --enable-sparse --with-colpack=`pwd`/ThirdParty/ColPack/ --prefix=`pwd`/inst
make
make install

