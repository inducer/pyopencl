#! /bin/bash

# https://github.com/conda-forge/intel-compiler-repack-feedstock/issues/7
sed -i 's/- pocl/- intel-opencl-rt!=2022.2/g' "$CONDA_ENVIRONMENT"
