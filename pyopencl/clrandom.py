from __future__ import annotations


__copyright__ = "Copyright (C) 2009-16 Andreas Kloeckner"

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


# {{{ documentation

__doc__ = """
PyOpenCL includes and uses some of the `Random123 random number generators
<https://www.deshawresearch.com/resources.html>`__ by D.E. Shaw
Research. In addition to being usable through the convenience functions above,
they are available in any piece of code compiled through PyOpenCL by::

    #include <pyopencl-random123/philox.cl>
    #include <pyopencl-random123/threefry.cl>

See the `Philox source
<https://github.com/inducer/pyopencl/blob/main/pyopencl/cl/pyopencl-random123/philox.cl>`__
and the `Threefry source
<https://github.com/inducer/pyopencl/blob/main/pyopencl/cl/pyopencl-random123/threefry.cl>`__
for some documentation if you're planning on using Random123 directly.

.. autoclass:: PhiloxGenerator

.. autoclass:: ThreefryGenerator

.. autofunction:: rand
.. autofunction:: fill_rand

"""

# }}}

import numpy as np

from pytools import memoize_method

import pyopencl as cl
import pyopencl.array as cl_array
import pyopencl.cltypes as cltypes
from pyopencl.tools import first_arg_dependent_memoize


# {{{ Random123 generators

class Random123GeneratorBase:
    """
    .. versionadded:: 2016.2

    .. automethod:: fill_uniform
    .. automethod:: uniform
    .. automethod:: fill_normal
    .. automethod:: normal
    """

    @property
    def header_name(self):
        raise NotImplementedError

    @property
    def generator_name(self):
        raise NotImplementedError

    @property
    def key_length(self):
        raise NotImplementedError

    def __init__(self, context, key=None, counter=None, seed=None):
        int32_info = np.iinfo(np.int32)
        from random import Random

        rng = Random(seed)

        if key is not None and counter is not None and seed is not None:
            raise TypeError("seed is unused and may not be specified "
                    "if both counter and key are given")

        if key is None:
            key = [
                    rng.randrange(
                        int(int32_info.min), int(int32_info.max)+1)
                    for i in range(self.key_length-1)]
        if counter is None:
            counter = [
                    rng.randrange(
                        int(int32_info.min), int(int32_info.max)+1)
                    for i in range(4)]

        self.context = context
        self.key = key
        self.counter = counter

        self.counter_max = int32_info.max

    @memoize_method
    def get_gen_kernel(self, dtype, distribution):
        size_multiplier = 1
        arg_dtype = dtype

        rng_key = (distribution, dtype)

        if rng_key in [("uniform", np.float64), ("normal", np.float64)]:
            c_type = "double"
            scale1_const = "((double) %r)" % (1/2**32)
            scale2_const = "((double) %r)" % (1/2**64)
            if distribution == "normal":
                transform = "box_muller"
            else:
                transform = ""

            rng_expr = (
                    "shift + scale * "
                    "%s( %s * convert_double4(gen)"
                    "+ %s * convert_double4(gen))"
                    % (transform, scale1_const, scale2_const))

            counter_multiplier = 2

        elif rng_key in [(dist, cmp_dtype)
                for dist in ["normal", "uniform"]
                for cmp_dtype in [
                    np.float32,
                    cltypes.float2,
                    cltypes.float3,
                    cltypes.float4,
                    ]]:
            c_type = "float"
            scale_const = "((float) %r)" % (1/2**32)

            if distribution == "normal":
                transform = "box_muller"
            else:
                transform = ""

            rng_expr = (
                    "shift + scale * %s(%s * convert_float4(gen))"
                    % (transform, scale_const))
            counter_multiplier = 1
            arg_dtype = np.float32
            try:
                _, size_multiplier = cltypes.vec_type_to_scalar_and_count[dtype]
            except KeyError:
                pass

        elif rng_key == ("uniform", np.int32):
            c_type = "int"
            rng_expr = (
                    "shift + convert_int4((convert_long4(gen) * scale) / %s)"
                    % (str(2**32)+"l")
                    )
            counter_multiplier = 1

        elif rng_key == ("uniform", np.int64):
            c_type = "long"
            rng_expr = (
                    "shift"
                    "+ convert_long4(gen) * (scale/two32) "
                    "+ ((convert_long4(gen) * scale) / two32)"
                    .replace("two32", (str(2**32)+"l")))
            counter_multiplier = 2

        else:
            raise TypeError(
                    "unsupported RNG distribution/data type combination '%s/%s'"
                    % rng_key)

        kernel_name = f"rng_gen_{self.generator_name}_{distribution}"
        src = """//CL//
            #include <{header_name}>

            #ifndef M_PI
            #ifdef M_PI_F
            #define M_PI M_PI_F
            #else
            #define M_PI 3.14159265359f
            #endif
            #endif

            typedef {output_t} output_t;
            typedef {output_t}4 output_vec_t;
            typedef {gen_name}_ctr_t ctr_t;
            typedef {gen_name}_key_t key_t;

            uint4 gen_bits(key_t *key, ctr_t *ctr)
            {{
                union {{
                    ctr_t ctr_el;
                    uint4 vec_el;
                }} u;

                u.ctr_el = {gen_name}(*ctr, *key);
                if (++ctr->v[0] == 0)
                    if (++ctr->v[1] == 0)
                        ++ctr->v[2];

                return u.vec_el;
            }}

            #if {include_box_muller}
            output_vec_t box_muller(output_vec_t x)
            {{
                #define BOX_MULLER(I, COMPA, COMPB) \
                    output_t r##I = sqrt(-2*log(x.COMPA)); \
                    output_t c##I; \
                    output_t s##I = sincos((output_t) (2*M_PI) * x.COMPB, &c##I);

                BOX_MULLER(0, x, y);
                BOX_MULLER(1, z, w);
                return (output_vec_t) (r0*c0, r0*s0, r1*c1, r1*s1);
            }}
            #endif

            #define GET_RANDOM_NUM(gen) {rng_expr}

            kernel void {kernel_name}(
                int k1,
                #if {key_length} > 2
                int k2, int k3,
                #endif
                int c0, int c1, int c2, int c3,
                global output_t *output,
                long out_size,
                output_t scale,
                output_t shift)
            {{
                #if {key_length} == 2
                key_t k = {{{{get_global_id(0), k1}}}};
                #else
                key_t k = {{{{get_global_id(0), k1, k2, k3}}}};
                #endif

                ctr_t c = {{{{c0, c1, c2, c3}}}};

                // output bulk
                unsigned long idx = get_global_id(0)*4;
                while (idx + 4 < out_size)
                {{
                    output_vec_t ran = GET_RANDOM_NUM(gen_bits(&k, &c));
                    vstore4(ran, 0, &output[idx]);
                    idx += 4*get_global_size(0);
                }}

                // output tail
                output_vec_t tail_ran = GET_RANDOM_NUM(gen_bits(&k, &c));
                if (idx < out_size)
                  output[idx] = tail_ran.x;
                if (idx+1 < out_size)
                  output[idx+1] = tail_ran.y;
                if (idx+2 < out_size)
                  output[idx+2] = tail_ran.z;
                if (idx+3 < out_size)
                  output[idx+3] = tail_ran.w;
            }}
            """.format(
                kernel_name=kernel_name,
                gen_name=self.generator_name,
                header_name=self.header_name,
                output_t=c_type,
                key_length=self.key_length,
                include_box_muller=int(distribution == "normal"),
                rng_expr=rng_expr
                )

        prg = cl.Program(self.context, src).build()
        knl = getattr(prg, kernel_name)
        knl.set_scalar_arg_dtypes(
                [np.int32] * (self.key_length - 1 + 4)
                + [None, np.int64, arg_dtype, arg_dtype])

        return knl, counter_multiplier, size_multiplier

    def _fill(self, distribution, ary, scale, shift, queue=None):
        """Fill *ary* with uniformly distributed random numbers in the interval
        *(a, b)*, endpoints excluded.

        :return: a :class:`pyopencl.Event`
        """

        if queue is None:
            queue = ary.queue

        knl, counter_multiplier, size_multiplier = \
                self.get_gen_kernel(ary.dtype, distribution)

        args = self.key + self.counter + [
                ary.data, ary.size*size_multiplier,
                scale, shift]

        n = ary.size
        from pyopencl.array import _splay
        gsize, lsize = _splay(queue.device, ary.size)

        evt = knl(queue, gsize, lsize, *args)
        ary.add_event(evt)

        self.counter[0] += n * counter_multiplier
        c1_incr, self.counter[0] = divmod(self.counter[0], self.counter_max)
        if c1_incr:
            self.counter[1] += c1_incr
            c2_incr, self.counter[1] = divmod(self.counter[1], self.counter_max)
            self.counter[2] += c2_incr

        return evt

    def fill_uniform(self, ary, a=0, b=1, queue=None):
        return self._fill("uniform", ary,
                scale=(b-a), shift=a, queue=queue)

    def uniform(self, *args, **kwargs):
        """Make a new empty array, apply :meth:`fill_uniform` to it.
        """
        a = kwargs.pop("a", 0)
        b = kwargs.pop("b", 1)

        result = cl_array.empty(*args, **kwargs)
        self.fill_uniform(result, queue=result.queue, a=a, b=b)
        return result

    def fill_normal(self, ary, mu=0, sigma=1, queue=None):
        """Fill *ary* with normally distributed numbers with mean *mu* and
        standard deviation *sigma*.
        """

        return self._fill("normal", ary, scale=sigma, shift=mu, queue=queue)

    def normal(self, *args, **kwargs):
        """Make a new empty array, apply :meth:`fill_normal` to it.
        """
        mu = kwargs.pop("mu", 0)
        sigma = kwargs.pop("sigma", 1)

        result = cl_array.empty(*args, **kwargs)
        self.fill_normal(result, queue=result.queue, mu=mu, sigma=sigma)
        return result


class PhiloxGenerator(Random123GeneratorBase):
    __doc__ = Random123GeneratorBase.__doc__

    header_name = "pyopencl-random123/philox.cl"
    generator_name = "philox4x32"
    key_length = 2


class ThreefryGenerator(Random123GeneratorBase):
    __doc__ = Random123GeneratorBase.__doc__

    header_name = "pyopencl-random123/threefry.cl"
    generator_name = "threefry4x32"
    key_length = 4

# }}}


@first_arg_dependent_memoize
def _get_generator(context):
    if context.devices[0].type & cl.device_type.CPU:
        gen = PhiloxGenerator(context)
    else:
        gen = ThreefryGenerator(context)

    return gen


def fill_rand(result, queue=None, a=0, b=1):
    """Fill *result* with random values in the range :math:`[0, 1)`.
    """
    if queue is None:
        queue = result.queue
    gen = _get_generator(queue.context)
    gen.fill_uniform(result, a=a, b=b)


def rand(queue, shape, dtype, luxury=None, a=0, b=1):
    """Return an array of *shape* filled with random values of *dtype*
    in the range :math:`[a, b)`.
    """

    if luxury is not None:
        from warnings import warn
        warn("Specifying the 'luxury' argument is deprecated and will stop being "
                "supported in PyOpenCL 2018.x", stacklevel=2)

    from pyopencl.array import Array
    gen = _get_generator(queue.context)
    result = Array(queue, shape, dtype)
    gen.fill_uniform(result, a=a, b=b)
    return result


# vim: filetype=pyopencl:foldmethod=marker
