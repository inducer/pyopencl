#!/bin/bash
set -e -x -o pipefail

if [[ ! -d ~/deps ]]; then

    # dependencies need to be setup

    mkdir -p ~/deps
    pushd ~/deps

    curl https://tiker.net/tmp/.tmux.conf
    yum install -y git yum openssl-devel ruby

    git clone --branch v2.3.0 https://github.com/OCL-dev/ocl-icd
    cd ocl-icd
    curl -L -O https://raw.githubusercontent.com/conda-forge/ocl-icd-feedstock/e2c03e3ddb1ff86630ccf80dc7b87a81640025ea/recipe/install-headers.patch
    git apply install-headers.patch
    curl -L -O https://github.com/isuruf/ocl-icd/commit/3862386b51930f95d9ad1089f7157a98165d5a6b.patch
    git apply 3862386b51930f95d9ad1089f7157a98165d5a6b.patch
    autoreconf -i
    chmod +x configure
    ./configure --prefix=/usr
    make -j$(nproc)
    make install
    
    popd
fi

# Compile wheels
PYTHON_VERSION=$(python -c 'import platform; print(platform.python_version())')
NUMPY_VERSION_SPEC='=='
if [[ "${PYTHON_VERSION}" == '3.6'* ]]; then
    NUMPY_VERSION_SPEC="${NUMPY_VERSION_SPEC}1.11.3"
elif [[ "${PYTHON_VERSION}" == '3.7'* ]]; then
    NUMPY_VERSION_SPEC="${NUMPY_VERSION_SPEC}1.14.5"
elif [[ "${PYTHON_VERSION}" == '3.8'* ]]; then
    NUMPY_VERSION_SPEC="${NUMPY_VERSION_SPEC}1.17.3"
elif [[ "${PYTHON_VERSION}" == '3.9'* ]]; then
    NUMPY_VERSION_SPEC="${NUMPY_VERSION_SPEC}1.19.5"
else
    # Unknown python version, let it unpinned instead
    NUMPY_VERSION_SPEC=''
fi
# TODO: declear dependencies except numpy as build-time dependency as per PEP518
# Build with the oldest numpy available to be compatible with newer ones
pip install "numpy${NUMPY_VERSION_SPEC}" pybind11 mako

# For bundling license files
pip install delocate
