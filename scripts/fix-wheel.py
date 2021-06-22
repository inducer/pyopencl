import sys
import os
import shutil
from glob import glob

from delocate import wheeltools

def add_library(fname, paths):
    print('Processing', fname)
    with wheeltools.InWheel(fname, fname):
        libs_dir = os.path.join('pyopencl', '.libs')
        os.makedirs(libs_dir, exist_ok=True)
        for lib_path in paths:
            shutil.copy2(lib_path, libs_dir)

def main():
    fname = sys.argv[1]
    paths = sys.argv[2:]
    add_library(fname, paths)

if __name__ == '__main__':
    main()
