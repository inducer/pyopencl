import sys
import os.path
import shutil
from glob import glob

from delocate import wheeltools

def add_library(paths):
    wheel_fnames = glob('/io/wheelhouse/pyopencl*.whl')
    for fname in wheel_fnames:
        print('Processing', fname)
        with wheeltools.InWheel(fname, fname):
            for lib_path in paths:
                shutil.copy2(lib_path, os.path.join('pyopencl', '.libs'))

def main():
    args = list(sys.argv)
    args.pop(0)
    add_library(args)

if __name__ == '__main__':
    main()
