from __future__ import division
from __future__ import absolute_import
import six

__copyright__ = """
Copyright (C) 2014 Andreas Kloeckner
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""


import numpy as np
import pyopencl as cl
from pyopencl.tools import bitlog2


# {{{ allocators

class AllocatorBase(object):
    def __call__(self, nbytes):
        try_count = 0

        while try_count < 2:
            try:
                return self.allocate(nbytes)
            except cl.Error as e:
                if not e.is_out_of_memory():
                    raise
                try_count += 1
                if try_count == 2:
                    raise

            self.try_release_blocks()

    def try_release_blocks(self):
        import gc
        gc.collect()

    def free(self, buf):
        buf.release()


class DeferredAllocator(AllocatorBase):
    is_deferred = True

    def __init__(self, context, mem_flags=cl.mem_flags.READ_WRITE):
        self.context = context
        self.mem_flags = mem_flags

    def allocate(self, nbytes):
        return cl.Buffer(self.context, self.mem_flags, nbytes)


_zero = np.array([0, 0, 0, 0], dtype=np.int8)


class ImmediateAllocator(AllocatorBase):
    is_deferred = False

    def __init__(self, queue, mem_flags=cl.mem_flags.READ_WRITE):
        self.context = queue.context
        self.queue = queue
        self.mem_flags = mem_flags

    def allocate(self, nbytes):
        buf = cl.Buffer(self.context, self.mem_flags, nbytes)

        # Make sure the buffer gets allocated right here and right now.
        # This looks (and is) expensive. But immediate allocators
        # have their main use in memory pools, whose basic assumption
        # is that allocation is too expensive anyway--but they rely
        # on exact 'out-of-memory' information.

        from pyopencl.cffi_cl import _enqueue_write_buffer
        _enqueue_write_buffer(
                self.queue, buf,
                _zero[:min(len(_zero), nbytes)],
                is_blocking=False)

        # No need to wait for completion here. clWaitForEvents (e.g.)
        # cannot return mem object allocation failures. This implies that
        # the buffer is faulted onto the device on enqueue.

        return buf

# }}}


# {{{ memory pool

class MemoryPool(object):
    mantissa_bits = 2
    mantissa_mask = (1 << mantissa_bits) - 1

    def __init__(self, allocator):
        self.allocator = allocator

        self.bin_nr_to_bin = {}

        if self.allocator.is_deferred:
            from warnings import warn
            warn("Memory pools expect non-deferred "
                    "semantics from their allocators. You passed a deferred "
                    "allocator, i.e. an allocator whose allocations can turn out to "
                    "be unavailable long after allocation.", statcklevel=2)

        self.active_blocks = 0

        self.stop_holding_flag = False

    @classmethod
    def bin_number(cls, size):
        l = bitlog2(size)

        mantissa_bits = cls.mantissa_bits
        if l >= mantissa_bits:
            shifted = size >> (l - mantissa_bits)
        else:
            shifted = size << (mantissa_bits - l)

        assert not (size and (shifted & (1 << mantissa_bits)) == 0)

        chopped = shifted & cls.mantissa_mask

        return l << mantissa_bits | chopped

    @classmethod
    def alloc_size(cls, bin_nr):
        mantissa_bits = cls.mantissa_bits

        exponent = bin_nr >> mantissa_bits
        mantissa = bin_nr & cls.mantissa_mask

        exp_minus_mbits = exponent-mantissa_bits
        if exp_minus_mbits >= 0:
            ones = (1 << exp_minus_mbits) - 1
            head = ((1 << mantissa_bits) | mantissa) << exp_minus_mbits
        else:
            ones = 0
            head = ((1 << mantissa_bits) | mantissa) >> -exp_minus_mbits

        assert not (ones & head)
        return head | ones

    def stop_holding(self):
        self.stop_holding_flag = True
        self.free_held()

    def free_held(self):
        for bin_nr, bin_list in six.iteritems(self.bin_nr_to_bin):
            while bin_list:
                self.allocator.free(bin_list.pop())

    @property
    def held_blocks(self):
        return sum(
                len(bin_list)
                for bin_list in six.itervalues(self.bin_nr_to_bin))

    def allocate(self, size):
        bin_nr = self.bin_number(size)
        bin_list = self.bin_nr_to_bin.setdefault(bin_nr, [])

        alloc_sz = self.alloc_size(bin_nr)

        if bin_list:
            # if (m_trace)
            #   std::cout
            #     << "[pool] allocation of size " << size
            #     << " served from bin " << bin_nr
            #     << " which contained " << bin_list.size()
            #     << " entries" << std::endl;
            self.active_blocks += 1
            return PooledBuffer(self, bin_list.pop(), alloc_sz)

        assert self.bin_number(alloc_sz) == bin_nr

        # if (m_trace)
        #   std::cout << "[pool] allocation of size " << size
        #   << " required new memory" << std::endl;

        try:
            result = self.allocator(alloc_sz)
            self.active_blocks += 1
            return PooledBuffer(self, result, alloc_sz)
        except cl.MemoryError:
            pass

        # if (m_trace)
        #   std::cout << "[pool] allocation triggered OOM, running GC" << std::endl;

        self.allocator.try_release_blocks()

        if bin_list:
            return bin_list.pop()

        # if (m_trace)
        #   std::cout << "[pool] allocation still OOM after GC" << std::endl;

        for _ in self._try_to_free_memory():
            try:
                result = self.allocator(alloc_sz)
                self.active_blocks += 1
                return PooledBuffer(self, result, alloc_sz)
            except cl.MemoryError:
                pass

        raise cl.MemoryError(
                "failed to free memory for allocation",
                routine="memory_pool::allocate",
                code=cl.status_code.MEM_OBJECT_ALLOCATION_FAILURE)

    __call__ = allocate

    def free(self, buf, size):
        self.active_blocks -= 1
        bin_nr = self.bin_number(size)

        if not self.stop_holding_flag:
            self.bin_nr_to_bin.setdefault(bin_nr, []).append(buf)

            # if (m_trace)
            #   std::cout << "[pool] block of size " << size << " returned to bin "
            #     << bin_nr << " which now contains " << get_bin(bin_nr).size()
            #     << " entries" << std::endl;
        else:
            self.allocator.free(buf)

    def _try_to_free_memory(self):
        for bin_nr, bin_list in six.iteritems(self.bin_nr_to_bin):
            while bin_list:
                self.allocator.free(bin_list.pop())
                self.held_blocks -= 1
                yield


class PooledBuffer(cl.MemoryObjectHolder):
    _id = 'buffer'

    def __init__(self, pool, buf, alloc_sz):
        self.pool = pool
        self.buf = buf
        self.ptr = buf.ptr
        self._alloc_sz = alloc_sz

    def release(self):
        self.pool.free(self.buf, self._alloc_sz)
        self.buf = None
        self.ptr = None

    def __del__(self):
        if self.buf is not None:
            self.release()

# }}}


# vim: foldmethod=marker
