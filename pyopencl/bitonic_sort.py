from __future__ import division, with_statement, absolute_import, print_function

__copyright__ = """
Copyright (c) 2011, Eric Bainville
Copyright (c) 2015, Ilya Efimoff
All rights reserved.
"""

# based on code at

__license__ = """
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors
may be used to endorse or promote products derived from this software without
specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import pyopencl as cl
from pyopencl.tools import dtype_to_ctype
from mako.template import Template
from operator import mul
from functools import reduce
from pytools import memoize_method


class BitonicSort(object):
    def __init__(self, context, shape, key_dtype, idx_dtype=None, axis=0):
        import pyopencl.bitonic_sort_templates as tmpl

        self.cached_defs = {}
        self.kernels_srcs = {
                'B2': tmpl.ParallelBitonic_B2,
                'B4': tmpl.ParallelBitonic_B4,
                'B8': tmpl.ParallelBitonic_B8,
                'B16': tmpl.ParallelBitonic_B16,
                'C4': tmpl.ParallelBitonic_C4,
                'BL': tmpl.ParallelBitonic_Local,
                'BLO': tmpl.ParallelBitonic_Local_Optim,
                'PML': tmpl.ParallelMerge_Local
                }

        self.dtype = dtype_to_ctype(key_dtype)
        self.context = context
        self.axis = axis
        if idx_dtype is None:
            self.argsort = 0
            self.idx_t = 'uint'  # Dummy
        else:
            self.argsort = 1
            self.idx_t = dtype_to_ctype(idx_dtype)
        self.defstpl = Template(tmpl.defines)
        self.rq = self.sort_b_prepare_wl(shape, self.axis)

    def __call__(self, _arr, idx=None, mkcpy=True):
        arr = _arr.copy() if mkcpy else _arr
        rq = self.rq
        p, nt, wg, aux = rq[0]
        if self.argsort and not type(idx)==type(None):
            if aux:
                p.run(arr.queue, (nt,), wg, arr.data, idx.data, cl.LocalMemory(wg[0]*4*arr.dtype.itemsize),\
                                                                cl.LocalMemory(wg[0]*4*idx.dtype.itemsize))
            for p, nt, wg,_ in rq[1:]:
                p.run(arr.queue, (nt,), wg, arr.data, idx.data)
        elif self.argsort==0:
            if aux:
                p.run(arr.queue, (nt,), wg, arr.data, cl.LocalMemory(wg[0]*4*arr.dtype.itemsize))
            for p, nt, wg,_ in rq[1:]:
                p.run(arr.queue, (nt,), wg, arr.data)
        else:
            raise ValueError("Array of indexes required for this sorter. If argsort is not needed,\
                              recreate sorter witout index datatype provided.")
        return arr

    @memoize_method
    def get_program(self, letter, params):
        if params in self.cached_defs.keys():
            defs = self.cached_defs[params]
        else:
            defs = self.defstpl.render(
                    NS="\\", argsort=self.argsort, inc=params[0], dir=params[1],
                    dtype=params[2], idxtype=params[3],
                    dsize=params[4], nsize=params[5])

            self.cached_defs[params] = defs
        kid = Template(self.kernels_srcs[letter]).render(argsort=self.argsort)
        prg = cl.Program(self.context, defs + kid).build()
        return prg

    def sort_b_prepare_wl(self, shape, axis):
        run_queue = []
        ds = int(shape[axis])
        size = reduce(mul, shape)
        ndim = len(shape)

        ns = reduce(mul, shape[(axis+1):]) if axis < ndim-1 else 1

        ds = int(shape[axis])
        allowb4 = True
        allowb8 = True
        allowb16 = True

        wg = min(ds, self.context.devices[0].max_work_group_size)
        length = wg >> 1
        prg = self.get_program('BLO', (1, 1, self.dtype, self.idx_t, ds, ns))
        run_queue.append((prg, size, (wg,), True))

        while length < ds:
            inc = length
            while inc > 0:
                ninc = 0
                direction = length << 1
                if allowb16 and inc >= 8 and ninc == 0:
                    letter = 'B16'
                    ninc = 4
                elif allowb8 and inc >= 4 and ninc == 0:
                    letter = 'B8'
                    ninc = 3
                elif allowb4 and inc >= 2 and ninc == 0:
                    letter = 'B4'
                    ninc = 2
                elif inc >= 0:
                    letter = 'B2'
                    ninc = 1

                nthreads = size >> ninc

                prg = self.get_program(letter,
                        (inc, direction, self.dtype, self.idx_t,  ds, ns))
                run_queue.append((prg, nthreads, None, False,))
                inc >>= ninc

            length <<= 1

        return run_queue
