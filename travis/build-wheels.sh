#!/bin/bash
set -e -x

cd /io
mkdir -p deps
cd deps

yum install -y git cmake yum wget
wget http://cache.ruby-lang.org/pub/ruby/2.1/ruby-2.1.2.tar.gz
tar -xvf ruby-2.1.2.tar.gz
cd ruby-2.1.2
./configure
make -j4
make install
cd ..
rm -rf ruby-2.1.2

git clone --branch v2.2.12 https://github.com/OCL-dev/ocl-icd
cd ocl-icd
wget https://raw.githubusercontent.com/conda-forge/ocl-icd-feedstock/master/recipe/install-headers.patch --no-check-certificate
git apply install-headers.patch
autoreconf -i
chmod +x configure
./configure --prefix=/usr
make -j4
make install

cd /io

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    "${PYBIN}/pip" install numpy pybind11 mako
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/pyopencl*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Bundle license files
/opt/python/cp37-cp37m/bin/pip install delocate
/opt/python/cp37-cp37m/bin/python /io/travis/fix-wheel.py /io/deps/ocl-icd/COPYING

/opt/python/cp37-cp37m/bin/pip install twine
for WHEEL in /io/wheelhouse/pyopencl*.whl; do
    # dev
    # /opt/python/cp37-cp37m/bin/twine upload \
    #     --skip-existing \
    #     --repository-url https://test.pypi.org/legacy/ \
    #     -u "${TWINE_USERNAME}" -p "${TWINE_PASSWORD}" \
    #     "${WHEEL}"
    # prod
    /opt/python/cp37-cp37m/bin/twine upload \
        --skip-existing \
        -u "${TWINE_USERNAME}" -p "${TWINE_PASSWORD}" \
        "${WHEEL}"
done
