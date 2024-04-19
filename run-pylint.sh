#!/bin/bash

set -o errexit -o nounset

ci_support="https://gitlab.tiker.net/inducer/ci-support/raw/main"

if [[ ! -f .pylintrc.yml ]]; then
    curl -o .pylintrc.yml "${ci_support}/.pylintrc-default.yml"
fi


if [[ ! -f .run-pylint.py ]]; then
    curl -L -o .run-pylint.py "${ci_support}/run-pylint.py"
fi


PYLINT_RUNNER_ARGS="--jobs=4 --yaml-rcfile=.pylintrc.yml"

if [[ -f .pylintrc-local.yml ]]; then
    PYLINT_RUNNER_ARGS+=" --yaml-rcfile=.pylintrc-local.yml"
fi

python .run-pylint.py $PYLINT_RUNNER_ARGS $(basename $PWD) test/*.py "$@"
