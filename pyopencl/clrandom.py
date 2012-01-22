from __future__ import division
import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.tools import first_arg_dependent_memoize
from pytools import memoize_method

import numpy as np




class RanluxGenerator(object):
    def __init__(self, queue, num_work_items,
            luxury=None, seed=None, no_warmup=False,
            use_legacy_init=False, max_work_items=None):
        if luxury is None:
            luxury = 4

        if seed is None:
            from time import time
            seed = int(time()*1e6) % 2<<30

        self.context = queue.context
        self.luxury = luxury
        self.num_work_items = num_work_items

        from pyopencl.characterize import has_double_support
        self.support_double = has_double_support(queue.device)

        self.no_warmup = no_warmup
        self.use_legacy_init = use_legacy_init
        self.max_work_items = max_work_items

        src = """
            %(defines)s

            #include <pyopencl-ranluxcl.cl>

            kernel void init_ranlux(unsigned seeds, global ranluxcl_state_t *ranluxcltab)
            {
              if (get_global_id(0) < %(num_work_items)d)
                ranluxcl_initialization(seeds, ranluxcltab);
            }
            """ % {
                    "defines": self.generate_settings_defines(),
                    "num_work_items": num_work_items
                }
        prg = cl.Program(queue.context, src).build()

        # {{{ compute work group size

        wg_size = None

        import sys
        import platform
        if ("darwin" in sys.platform
                and "Apple" in queue.device.platform.vendor
                and platform.mac_ver()[0].startswith("10.7")
                and queue.device.type == cl.device_type.CPU):
            wg_size = (1,)

        self.wg_size = wg_size

        # }}}

        self.state = cl_array.empty(queue, (num_work_items, 112), dtype=np.uint8)
        self.state.fill(17)

        prg.init_ranlux(queue, (num_work_items,), self.wg_size, np.uint32(seed),
                self.state.data)

    def generate_settings_defines(self, include_double_pragma=True):
        lines = []
        if include_double_pragma and self.support_double:
            lines.append("#pragma OPENCL EXTENSION cl_khr_fp64 : enable")

        lines.append("#define RANLUXCL_LUX %d" % self.luxury)

        if self.no_warmup:
            lines.append("#define RANLUXCL_NO_WARMUP")

        if self.support_double:
            lines.append("#define RANLUXCL_SUPPORT_DOUBLE")

        if self.use_legacy_init:
            lines.append("#define RANLUXCL_USE_LEGACY_INITIALIZATION")

            if self.max_work_items:
                lines.append("#define RANLUXCL_MAXWORKITEMS %d" % self.max_work_items)

        return "\n".join(lines)

    @memoize_method
    def get_gen_kernel(self, dtype, flavor=""):
        size_multiplier = 1
        arg_dtype = dtype

        if dtype == np.float64:
            bits = 64
            c_type = "double"
            rng_expr = "(shift + scale * gen)"
        elif dtype == np.float32:
            bits = 32
            c_type = "float"
            rng_expr = "(shift + scale * gen)"
        elif dtype == cl_array.vec.float2:
            bits = 32
            c_type = "float"
            rng_expr = "(shift + scale * gen)"
            size_multiplier = 2
            arg_dtype = np.float32
        elif dtype in [cl_array.vec.float3, cl_array.vec.float4]:
            bits = 32
            c_type = "float"
            rng_expr = "(shift + scale * gen)"
            size_multiplier = 4
            arg_dtype = np.float32
        elif dtype == np.int32:
            assert flavor == ""
            bits = 32
            c_type = "int"
            rng_expr = ("(shift "
                    "+ convert_int4(scale * gen) "
                    "+ convert_int4((scale / (1<<24)) * gen))"
                    )
        else:
            raise TypeError("unsupported RNG data type '%s'" % dtype)

        rl_flavor = "%d%s" % (bits, flavor)

        src = """//CL//
            %(defines)s

            #include <pyopencl-ranluxcl.cl>

            typedef %(output_t)s output_t;
            typedef %(output_t)s4 output_vec_t;
            #define NUM_WORKITEMS %(num_work_items)d
            #define RANLUX_FUNC ranluxcl##%(rlflavor)s
            #define GET_RANDOM_NUM(gen) %(rng_expr)s

            kernel void generate(
                global ranluxcl_state_t *ranluxcltab,
                global output_t *output,
                unsigned long out_size,
                output_t scale,
                output_t shift)
            {

              ranluxcl_state_t ranluxclstate;
              ranluxcl_download_seed(&ranluxclstate, ranluxcltab);

              // output bulk
              unsigned long idx = get_global_id(0)*4;
              while (idx + 4 < out_size)
              {
                  vstore4(GET_RANDOM_NUM(RANLUX_FUNC(&ranluxclstate)), idx >> 2, output);
                  idx += 4*NUM_WORKITEMS;
              }

              // output tail
              output_vec_t tail_ran = GET_RANDOM_NUM(RANLUX_FUNC(&ranluxclstate));
              if (idx < out_size)
                output[idx] = tail_ran.x;
              if (idx+1 < out_size)
                output[idx+1] = tail_ran.y;
              if (idx+2 < out_size)
                output[idx+2] = tail_ran.z;
              if (idx+3 < out_size)
                output[idx+3] = tail_ran.w;

              ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
            """ % {
                "defines": self.generate_settings_defines(),
                "rlflavor": rl_flavor,
                "output_t": c_type,
                "num_work_items": self.num_work_items,
                "rng_expr": rng_expr
            }

        prg = cl.Program(self.context, src).build()
        knl = prg.generate
        knl.set_scalar_arg_dtypes([None, None, np.uint64, arg_dtype, arg_dtype])

        return knl, size_multiplier

    def fill_uniform(self, ary, a=0, b=1, queue=None):
        if queue is None:
            queue = ary.queue

        knl, size_multiplier = self.get_gen_kernel(ary.dtype, "")
        knl(queue,
                (self.num_work_items,), None,
                self.state.data, ary.data, ary.size*size_multiplier,
                b-a, a)

    def uniform(self, *args, **kwargs):
        a = kwargs.pop("a", 0)
        b = kwargs.pop("b", 1)

        result = cl_array.empty(*args, **kwargs)

        self.fill_uniform(result, queue=result.queue, a=a, b=b)
        return result

    def fill_normal(self, ary, mu=0, sigma=1, queue=None):
        if queue is None:
            queue = ary.queue

        knl, size_multiplier = self.get_gen_kernel(ary.dtype, "norm")
        knl(queue,
                (self.num_work_items,), self.wg_size,
                self.state.data, ary.data, ary.size*size_multiplier, sigma, mu)

    def normal(self, *args, **kwargs):
        mu = kwargs.pop("mu", 0)
        sigma = kwargs.pop("sigma", 1)

        result = cl_array.empty(*args, **kwargs)

        self.fill_normal(result, queue=result.queue, mu=mu, sigma=sigma)
        return result

    @memoize_method
    def get_sync_kernel(self):
        src = """//CL//
            %(defines)s

            #include <pyopencl-ranluxcl.cl>

            kernel void sync(
                global ranluxcl_state_t *ranluxcltab)
            {
              ranluxcl_state_t ranluxclstate;
              ranluxcl_download_seed(&ranluxclstate, ranluxcltab);
              ranluxcl_synchronize(&ranluxclstate);
              ranluxcl_upload_seed(&ranluxclstate, ranluxcltab);
            }
            """ % {
                "defines": self.generate_settings_defines(),
            }
        prg = cl.Program(self.context, src).build()
        return prg.sync

    def synchronize(self, queue):
        self.get_sync_kernel()(queue, (self.num_work_items,), self.wg_size, self.state.data)






@first_arg_dependent_memoize
def _get_generator(queue, luxury=None):
    if queue.device.type == cl.device_type.CPU:
        num_work_items = 8 * queue.device.max_compute_units
    else:
        num_work_items = 64 * queue.device.max_compute_units

    gen = RanluxGenerator(queue, num_work_items, luxury=luxury)
    queue.finish()
    return gen




def fill_rand(result, queue=None, luxury=4, a=0, b=1):
    if queue is None:
        queue = result.queue
    gen = _get_generator(queue, luxury=luxury)
    gen.fill_uniform(result, a=a, b=b)




def rand(*args, **kwargs):
    def inner_rand(queue, shape, dtype, luxury=None, a=0, b=1):
        from pyopencl.array import Array
        luxury = kwargs.pop("luxury", None)

        gen = _get_generator(queue, luxury)
        result = Array(queue, shape, dtype)
        gen.fill_uniform(result, a=a, b=b)
        return result

    if isinstance(args[0], cl.Context):
        from warnings import warn
        warn("Passing a context as first argument is deprecated. "
            "This will be continue to be accepted througout "
            "versions 2011.x of PyOpenCL.",
            DeprecationWarning, 2)
        args = args[1:]

    return inner_rand(*args, **kwargs)




# vim: filetype=pyopencl
