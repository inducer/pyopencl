#!/usr/bin/env bash

set -o xtrace

git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader
cd OpenCL-ICD-Loader
git checkout aec3952654832211636fc4af613710f80e203b0a
cd ..
git clone https://github.com/KhronosGroup/OpenCL-Headers
cd OpenCL-Headers
git checkout dcd5bede6859d26833cd85f0d6bbcee7382dc9b3
cd ..


cmake -D CMAKE_INSTALL_PREFIX=./OpenCL-Headers/install -S ./OpenCL-Headers -B ./OpenCL-Headers/build 
cmake --build ./OpenCL-Headers/build --target install

cmake -D CMAKE_PREFIX_PATH=${PWD}/OpenCL-Headers/install -D CMAKE_INSTALL_PREFIX=./OpenCL-ICD-Loader/install -S ./OpenCL-ICD-Loader -B ./OpenCL-ICD-Loader/build 
cmake --build ./OpenCL-ICD-Loader/build --target install --config Release
