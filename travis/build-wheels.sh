#!/bin/bash
set -e -x

mkdir -p /deps
cd /deps

function start_spinner {
    if [ -n "$SPINNER_PID" ]; then
        return
    fi

    >&2 echo "Building libraries..."
    # Start a process that runs as a keep-alive
    # to avoid travis quitting if there is no output
    (while true; do
        sleep 60
        >&2 echo "Still building..."
    done) &
    SPINNER_PID=$!
    disown
}

function stop_spinner {
    if [ ! -n "$SPINNER_PID" ]; then
        return
    fi

    kill $SPINNER_PID
    unset SPINNER_PID

    >&2 echo "Building libraries finished."
}

start_spinner

yum install -y git yum
curl -L -O http://cache.ruby-lang.org/pub/ruby/2.1/ruby-2.1.2.tar.gz
tar -xf ruby-2.1.2.tar.gz
cd ruby-2.1.2
./configure --disable-install-doc --disable-install-rdoc
make -j4
make install
cd ..
rm -rf ruby-2.1.2

git clone --branch v2.2.12 https://github.com/OCL-dev/ocl-icd
cd ocl-icd
curl -L -O https://raw.githubusercontent.com/conda-forge/ocl-icd-feedstock/22625432a0ae85920825dfeb103af9fe7bd6a950/recipe/install-headers.patch
git apply install-headers.patch
curl -L -O https://github.com/isuruf/ocl-icd/commit/3862386b51930f95d9ad1089f7157a98165d5a6b.patch
git apply 3862386b51930f95d9ad1089f7157a98165d5a6b.patch
autoreconf -i
chmod +x configure
./configure --prefix=/usr
make -j4
make install
cd ..

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    if [[ "${PYBIN}" == *cp36* ]]; then
        NUMPY_VERSION="1.11.3"
    elif [[ "${PYBIN}" == *cp37* ]]; then
        NUMPY_VERSION="1.14.5"
    elif [[ "${PYBIN}" == *cp38* ]]; then
        NUMPY_VERSION="1.17.3"
    else
        continue
    fi
    # Build with the oldest numpy available to be compatible with newer ones
    "${PYBIN}/pip" install "numpy==${NUMPY_VERSION}" pybind11 mako
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/ --no-deps
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/pyopencl*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/
done

# Bundle license files

/opt/python/cp37-cp37m/bin/pip install delocate
/opt/python/cp37-cp37m/bin/python /io/travis/fix-wheel.py /deps/ocl-icd/COPYING

if [[ "${TWINE_USERNAME}" == "" ]]; then
    echo "TWINE_USERNAME not set. Skipping uploading wheels"
    exit 0
fi

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

stop_spinner
