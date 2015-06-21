from __future__ import division
from __future__ import absolute_import
from __future__ import print_function
from six.moves import range

__copyright__ = "Copyright (C) 2009 Andreas Kloeckner"

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

import pyopencl as cl
import numpy as np




# {{{ timing helpers

class Timer:
    def __init__(self, queue):
        self.queue = queue

    def start(self):
        pass

    def stop(self):
        pass

    def add_event(self, evt):
        pass

    def get_elapsed(self):
        pass




class WallTimer(Timer):
    def start(self):
        from time import time
        self.queue.finish()
        self.start = time()

    def stop(self):
        from time import time
        self.queue.finish()
        self.end = time()

    def get_elapsed(self):
        return self.end-self.start




def _get_time(queue, f, timer_factory=None, desired_duration=0.1,
        warmup_rounds=3):

    if timer_factory is None:
        timer_factory = WallTimer

    count = 1

    while True:
        timer = timer_factory(queue)

        for i in range(warmup_rounds):
            f()
        warmup_rounds = 0

        timer.start()
        for i in range(count):
            timer.add_event(f())
        timer.stop()

        elapsed = timer.get_elapsed()
        if elapsed < desired_duration:
            if elapsed == 0:
                count *= 5
            else:
                new_count = int(desired_duration/elapsed)

                new_count = max(2*count, new_count)
                new_count = min(10*count, new_count)
                count = new_count

        else:
            return elapsed/count

# }}}




# {{{ transfer measurements

class HostDeviceTransferBase(object):
    def __init__(self, queue, block_size):
        self.queue = queue
        self.host_buf = np.empty(block_size, dtype=np.uint8)
        self.dev_buf = cl.Buffer(queue.context, cl.mem_flags.READ_WRITE, block_size)

class HostToDeviceTransfer(HostDeviceTransferBase):
    def do(self):
        return cl.enqueue_copy(self. queue, self.dev_buf, self.host_buf)

class DeviceToHostTransfer(HostDeviceTransferBase):
    def do(self):
        return cl.enqueue_copy(self. queue, self.host_buf, self.dev_buf)

class DeviceToDeviceTransfer(object):
    def __init__(self, queue, block_size):
        self.queue = queue
        self.dev_buf_1 = cl.Buffer(queue.context, cl.mem_flags.READ_WRITE, block_size)
        self.dev_buf_2 = cl.Buffer(queue.context, cl.mem_flags.READ_WRITE, block_size)

    def do(self):
        return cl.enqueue_copy(self. queue, self.dev_buf_2, self.dev_buf_1)

class HostToDeviceTransfer(HostDeviceTransferBase):
    def do(self):
        return cl.enqueue_copy(self. queue, self.dev_buf, self.host_buf)


def transfer_latency(queue, transfer_type, timer_factory=None):
    transfer = transfer_type(queue, 1)
    return _get_time(queue, transfer.do, timer_factory=timer_factory)

def transfer_bandwidth(queue, transfer_type, block_size, timer_factory=None):
    """Measures one-sided bandwidth."""

    transfer = transfer_type(queue, block_size)
    return block_size/_get_time(queue, transfer.do, timer_factory=timer_factory)

# }}}




def get_profiling_overhead(ctx, timer_factory=None):
    no_prof_queue = cl.CommandQueue(ctx)
    transfer = DeviceToDeviceTransfer(no_prof_queue, 1)
    no_prof_time = _get_time(no_prof_queue, transfer.do, timer_factory=timer_factory)

    prof_queue = cl.CommandQueue(ctx,
            properties=cl.command_queue_properties.PROFILING_ENABLE)
    transfer = DeviceToDeviceTransfer(prof_queue, 1)
    prof_time = _get_time(prof_queue, transfer.do, timer_factory=timer_factory)

    return prof_time - no_prof_time, prof_time

def get_empty_kernel_time(queue, timer_factory=None):
    prg = cl.Program(queue.context, """
        __kernel void empty()
        { }
        """).build()

    knl = prg.empty

    def f():
        knl(queue, (1,), None)

    return _get_time(queue, f, timer_factory=timer_factory)

def _get_full_machine_kernel_rate(queue, src, args, name="benchmark", timer_factory=None):
    prg = cl.Program(queue.context, src).build()

    knl = getattr(prg, name)

    dev = queue.device
    global_size = 4 * dev.max_compute_units
    def f():
        knl(queue, (global_size,), None, *args)

    rates = []
    num_dips = 0

    while True:
        elapsed = _get_time(queue, f, timer_factory=timer_factory)
        rate = global_size/elapsed
        print(global_size, rate, num_dips)

        keep_trying = not rates

        if rates and rate > 1.05*max(rates): # big improvement
            keep_trying = True
            num_dips = 0

        if rates and rate < 0.9*max(rates) and num_dips < 3: # big dip
            keep_trying = True
            num_dips += 1

        if keep_trying:
            global_size *= 2
            last_rate = rate
            rates.append(rate)
        else:
            rates.append(rate)
            return max(rates)

def get_add_rate(queue, type="float", timer_factory=None):
    return 50*10*_get_full_machine_kernel_rate(queue, """
        typedef %(op_t)s op_t;
        __kernel void benchmark()
        {
            local op_t tgt[1024];
            op_t val = get_global_id(0);

            for (int i = 0; i < 10; ++i)
            {
                val += val; val += val; val += val; val += val; val += val;
                val += val; val += val; val += val; val += val; val += val;

                val += val; val += val; val += val; val += val; val += val;
                val += val; val += val; val += val; val += val; val += val;

                val += val; val += val; val += val; val += val; val += val;
                val += val; val += val; val += val; val += val; val += val;

                val += val; val += val; val += val; val += val; val += val;
                val += val; val += val; val += val; val += val; val += val;

                val += val; val += val; val += val; val += val; val += val;
                val += val; val += val; val += val; val += val; val += val;
            }
            tgt[get_local_id(0)] = val;
        }
        """ % dict(op_t=type), ())




# vim: foldmethod=marker:filetype=pyopencl
