#! /usr/bin/env python


import pyopencl as cl
import sys

ctx = cl.create_some_context()
with open(sys.argv[1]) as inf:
    src = inf.read()

prg = cl.Program(ctx, src).build()

print(prg.binaries[0].decode())
