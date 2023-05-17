#! /usr/bin/env python

# Prints a string representation of the binary kernel compiled
# from the first command line argument.

import pyopencl as cl
import sys

ctx = cl.create_some_context()
with open(sys.argv[1]) as inf:
    src = inf.read()

prg = cl.Program(ctx, src).build()

print(''.join(map(chr, (prg.binaries[0]))))
