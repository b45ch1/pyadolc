#!/bin/bash

SYSTEM=`uname -m`
OLDDIR=`pwd`
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd $DIR

# download ADOL-C
git clone https://git.gitorious.org/adol-c/adol-c.git PACKAGES/ADOL-C
# git clone https://github.com/b45ch1/adol-c.git PACKAGES/ADOL-C
cd PACKAGES/ADOL-C
# git checkout e686fc236baf2c109f9c7f43da9bbd88b558f74d
git checkout 4f72634
cd $DIR

# download ColPack
wget -P PACKAGES wget http://cscapes.cs.purdue.edu/download/ColPack/ColPack-1.0.9.tar.gz

# build ColPack
cd $DIR/PACKAGES
tar xfvz ColPack-1.0.9.tar.gz
cd ColPack-1.0.9
if [ SYSTEM = "x86_64" ]; then
./configure --prefix=`pwd`/../ADOL-C/ThirdParty/ColPack/ --libdir='${prefix}/lib64'
else
./configure --prefix=`pwd`/../ADOL-C/ThirdParty/ColPack/ --libdir='${prefix}/lib'
fi
make
make install

# build ADOL-C
cd ../ADOL-C
./update_versions.sh
./configure --enable-sparse --with-colpack=`pwd`/ThirdParty/ColPack/ --prefix=`pwd`/inst
make
make install
