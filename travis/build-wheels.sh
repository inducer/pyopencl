#!/bin/bash
set -e -x

export PYHOME=/home
export CL_H=${PYHOME}/cl_h
export CL_ICDLOAD=${PYHOME}/cl_icdload

cd ${PYHOME}
yum install -y git cmake
git clone https://github.com/KhronosGroup/OpenCL-Headers.git ${CL_H}
git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader.git ${CL_ICDLOAD}
ln -s ${CL_H}/CL /usr/include/CL
make -C ${CL_ICDLOAD}
cp -r ${CL_ICDLOAD}/build/lib/lib* /usr/lib

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install numpy pybind11 mako
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done


/opt/python/cp37-cp37m/bin/pip install twine
for WHEEL in /io/wheelhouse/gmagno_pyopencl*.whl; do
    # /opt/python/cp37-cp37m/bin/twine upload --repository-url https://test.pypi.org/legacy/ "${WHEEL}"
    /opt/python/cp37-cp37m/bin/twine upload -u "${TWINE_USERNAME}" -p "${TWINE_PASSWORD}" "${WHEEL}"
done
