#!/usr/bin/env bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

set -e -x

mkdir -p ~/deps
cd ~/deps

git clone --branch v2.3.1 https://github.com/OCL-dev/ocl-icd
cd ocl-icd
curl -L -O https://raw.githubusercontent.com/conda-forge/ocl-icd-feedstock/e2c03e3ddb1ff86630ccf80dc7b87a81640025ea/recipe/install-headers.patch
git apply install-headers.patch
curl -L -O https://github.com/isuruf/ocl-icd/commit/307f2267100a2d1383f0c4a77344b127c0857588.patch
git apply 307f2267100a2d1383f0c4a77344b127c0857588.patch
autoreconf -i
chmod +x configure
./configure --prefix=/usr
make -j4
make install

# Bundle license files
echo "PyOpenCL wheel includes ocl-icd which is licensed as below" >> ${SCRIPT_DIR}/../LICENSE
cat ~/deps/ocl-icd/COPYING >> ${SCRIPT_DIR}/../LICENSE