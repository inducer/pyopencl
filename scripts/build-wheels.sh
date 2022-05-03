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

#start_spinner

git config --global --add safe.directory /io

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
make -j4
make install
cd ..

# Bundle license files
echo "PyOpenCL wheel includes ocl-icd which is licensed as below" >> /io/LICENSE
cat /deps/ocl-icd/COPYING >> /io/LICENSE

# Compile wheels
for PYBIN in /opt/python/*/bin; do
    # Build with the oldest numpy available to be compatible with newer ones
    "${PYBIN}/pip" install oldest-supported-numpy pybind11 mako
    "${PYBIN}/pip" wheel /io/ -w wheelhouse/ --no-deps
done

# Bundle external shared libraries into the wheels
for whl in wheelhouse/pyopencl*.whl; do
    auditwheel repair "$whl" -w /io/wheelhouse/ --lib-sdir=/.libs
done

if [[ "${TWINE_USERNAME}" == "" ]]; then
    echo "TWINE_USERNAME not set. Skipping uploading wheels"
    exit 0
fi

/opt/python/cp39-cp39/bin/pip install twine
for WHEEL in /io/wheelhouse/pyopencl*.whl; do
    # dev
    # /opt/python/cp39-cp39/bin/twine upload \
    #     --skip-existing \
    #     --repository-url https://test.pypi.org/legacy/ \
    #     -u "${TWINE_USERNAME}" -p "${TWINE_PASSWORD}" \
    #     "${WHEEL}"
    # prod
    /opt/python/cp39-cp39/bin/twine upload \
        --skip-existing \
        -u "${TWINE_USERNAME}" -p "${TWINE_PASSWORD}" \
        "${WHEEL}"
done

#stop_spinner
