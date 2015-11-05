#! /usr/bin/env python

from __future__ import division

import pyopencl as cl
import sys

ctx = cl.create_some_context()
with open(sys.argv[1], "r") as inf:
    src = inf.read()

prg = cl.Program(ctx, src).build()

print(prg.binaries[0].decode())
