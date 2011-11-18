"""Various helpful bits and pieces without much of a common theme."""

from __future__ import division

__copyright__ = "Copyright (C) 2010 Andreas Kloeckner"

__license__ = """
Permission is hereby granted, free of charge, to any person
obtaining a copy of this software and associated documentation
files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use,
copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""



import numpy as np
from decorator import decorator
import pyopencl as cl

from pyopencl.compyte.dtypes import (
        register_dtype, _fill_dtype_registry,
        dtype_to_ctype)

_fill_dtype_registry(respect_windows=True)



PooledBuffer = cl.PooledBuffer
CLAllocator = cl.CLAllocator
MemoryPool = cl.MemoryPool




first_arg_dependent_memoized_functions = []




@decorator
def first_arg_dependent_memoize(func, context, *args):
    """Provides memoization for things that get created inside
    a context, i.e. mainly programs and kernels. Assumes that
    the first argument of the decorated function is the context.
    """
    try:
        ctx_dict = func._pyopencl_first_arg_dep_memoize_dic
    except AttributeError:
        # FIXME: This may keep contexts alive longer than desired.
        # But I guess since the memory in them is freed, who cares.
        ctx_dict = func._pyopencl_first_arg_dep_memoize_dic = {}

    try:
        return ctx_dict[context][args]
    except KeyError:
        first_arg_dependent_memoized_functions.append(func)
        arg_dict = ctx_dict.setdefault(context, {})
        result = func(context, *args)
        arg_dict[args] = result
        return result

context_dependent_memoize = first_arg_dependent_memoize




def clear_context_caches():
    for func in first_arg_dependent_memoized_functions:
        try:
            ctx_dict = func._pyopencl_first_arg_dep_memoize_dic
        except AttributeError:
            pass
        else:
            ctx_dict.clear()

import atexit
atexit.register(clear_context_caches)




def pytest_generate_tests_for_pyopencl(metafunc):
    class ContextFactory:
        def __init__(self, device):
            self.device = device

        def __call__(self):
            # Get rid of leftovers from past tests.
            # CL implementations are surprisingly limited in how many
            # simultaneous contexts they allow...

            clear_context_caches()

            from gc import collect
            collect()

            return cl.Context([self.device])

        def __str__(self):
            return "<context factory for %s>" % self.device

    platform_blacklist = []
    device_blacklist = []

    import os
    bl_string = os.environ.get("PYOPENCL_TEST_PLATFORM_BLACKLIST", "")
    if bl_string:
        platform_blacklist = [s.lower() for s in bl_string.split(",")]
    bl_string = os.environ.get("PYOPENCL_TEST_DEVICE_BLACKLIST", "")
    if bl_string:
        for s in bl_string.split(","):
            vendor_name, device_name = s.split(":")
            device_blacklist.append((vendor_name.lower(), device_name.lower()))

    from pytools import any

    if ("device" in metafunc.funcargnames
            or "ctx_factory" in metafunc.funcargnames
            or "ctx_getter" in metafunc.funcargnames):
        arg_dict = {}

        for platform in cl.get_platforms():
            if "platform" in metafunc.funcargnames:
                arg_dict["platform"] = platform

            platform_ish = (platform.name + ' ' + platform.vendor).lower()
            if any(bl_plat in platform_ish for bl_plat in platform_blacklist):
                continue

            for device in platform.get_devices():

                device_ish = (device.name + ' ' + platform.vendor).lower()
                if any(bl_plat in platform_ish and bl_dev in device_ish
                        for bl_plat, bl_dev in device_blacklist):
                    continue

                if "device" in metafunc.funcargnames:
                    arg_dict["device"] = device

                if "ctx_factory" in metafunc.funcargnames:
                    arg_dict["ctx_factory"] = ContextFactory(device)

                if "ctx_getter" in metafunc.funcargnames:
                    from warnings import warn
                    warn("The 'ctx_getter' arg is deprecated in favor of 'ctx_factory'.",
                            DeprecationWarning)
                    arg_dict["ctx_getter"] = ContextFactory(device)

                metafunc.addcall(funcargs=arg_dict.copy(),
                        id=", ".join("%s=%s" % (arg, value)
                                for arg, value in arg_dict.iteritems()))

    elif "platform" in metafunc.funcargnames:
        for platform in cl.get_platforms():

            platform_ish = (platform.name + ' ' + platform.vendor).lower()
            if any(bl_plat in platform_ish for bl_plat in platform_blacklist):
                continue

            metafunc.addcall(
                    funcargs=dict(platform=platform),
                    id=str(platform))




# {{{ C argument lists

class Argument:
    def __init__(self, dtype, name):
        self.dtype = np.dtype(dtype)
        self.name = name

    def __repr__(self):
        return "%s(%r, %s, %d)" % (
                self.__class__.__name__,
                self.name,
                self.dtype)

class VectorArg(Argument):
    def declarator(self):
        return "__global %s *%s" % (dtype_to_ctype(self.dtype), self.name)

class ScalarArg(Argument):
    def declarator(self):
        return "%s %s" % (dtype_to_ctype(self.dtype), self.name)





def parse_c_arg(c_arg):
    c_arg = (c_arg
            .replace("__global", "")
            .replace("__local", "")
            .replace("__constant", ""))

    from pyopencl.compyte.dtypes import parse_c_arg_backend
    return parse_c_arg_backend(c_arg, ScalarArg, VectorArg)

# }}}




def get_gl_sharing_context_properties():
    ctx_props = cl.context_properties

    from OpenGL import platform as gl_platform, GLX, WGL

    props = []

    import sys
    if sys.platform == "linux2":
        props.append(
            (ctx_props.GL_CONTEXT_KHR, gl_platform.GetCurrentContext()))
        props.append(
                (ctx_props.GLX_DISPLAY_KHR, 
                    GLX.glXGetCurrentDisplay()))
    elif sys.platform == "win32":
        props.append(
            (ctx_props.GL_CONTEXT_KHR, gl_platform.GetCurrentContext()))
        props.append(
                (ctx_props.WGL_HDC_KHR, 
                    WGL.wglGetCurrentDC()))
    elif sys.platform == "darwin":
        props.append(
            (ctx_props.CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE, cl.get_apple_cgl_share_group()))
    else:
        raise NotImplementedError("platform '%s' not yet supported" 
                % sys.platform)

    return props



# vim: foldmethod=marker
