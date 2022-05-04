#!/usr/bin/env bash

set -o xtrace

git clone --branch v2022.01.04 https://github.com/KhronosGroup/OpenCL-ICD-Loader
git clone --branch v2022.01.04 https://github.com/KhronosGroup/OpenCL-Headers



cmake -D CMAKE_INSTALL_PREFIX=./OpenCL-Headers/install -S ./OpenCL-Headers -B ./OpenCL-Headers/build 
cmake --build ./OpenCL-Headers/build --target install

cmake -D CMAKE_PREFIX_PATH=${PWD}/OpenCL-Headers/install -D OPENCL_ICD_LOADER_HEADERS_DIR=${PWD}/OpenCL-Headers/install/include -D CMAKE_INSTALL_PREFIX=./OpenCL-ICD-Loader/install -S ./OpenCL-ICD-Loader -B ./OpenCL-ICD-Loader/build 
cmake --build ./OpenCL-ICD-Loader/build --target install --config Release
