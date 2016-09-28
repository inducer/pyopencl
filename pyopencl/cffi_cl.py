from __future__ import division, absolute_import

__copyright__ = """
Copyright (C) 2013 Marko Bencun
Copyright (C) 2014 Andreas Kloeckner
Copyright (C) 2014 Yichao Yu
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

import six
from six.moves import map, range, zip, intern

import warnings
from warnings import warn
import numpy as np
import sys

from pytools import memoize_method

from pyopencl._cffi import ffi as _ffi
from .compyte.array import f_contiguous_strides, c_contiguous_strides


from pyopencl._cffi import lib as _lib


class _CLKernelArg(object):
    pass


# {{{ hook up connections between the wrapper and the interperter

import gc
_py_gc = _ffi.callback('int(void)')(gc.collect)

_pyrefs = {}


@_ffi.callback('void(void*)')
def _py_deref(handle):
    try:
        del _pyrefs[handle]
    except:
        pass


# return a new reference of the object pointed to by the handle.
# The return value might be different with the input (on PyPy).
# _py_deref should be called (once) when the object is not needed anymore.
@_ffi.callback('void*(void*)')
def _py_ref(handle):
    obj = _ffi.from_handle(handle)
    handle = _ffi.new_handle(obj)
    _pyrefs[handle] = handle
    return handle


@_ffi.callback('void(void*, cl_int)')
def _py_call(handle, status):
    _ffi.from_handle(handle)(status)


_lib.set_py_funcs(_py_gc, _py_ref, _py_deref, _py_call)

# }}}


# {{{ compatibility shims

# are we running on pypy?
_PYPY = '__pypy__' in sys.builtin_module_names
_CPY2 = not _PYPY and sys.version_info < (3,)
_CPY26 = _CPY2 and sys.version_info < (2, 7)

try:
    _unicode = eval('unicode')
    _ffi_pystr = _ffi.string
except:
    _unicode = str
    _bytes = bytes

    def _ffi_pystr(s):
        return _ffi.string(s).decode() if s else None
else:
    try:
        _bytes = bytes
    except:
        _bytes = str


def _to_cstring(s):
    if isinstance(s, _unicode):
        return s.encode()
    return s

# }}}


# {{{ wrapper tools

# {{{ _CArray helper classes

class _CArray(object):
    def __init__(self, ptr):
        self.ptr = ptr
        self.size = _ffi.new('uint32_t*')

    def __del__(self):
        if self.ptr != _ffi.NULL:
            _lib.free_pointer(self.ptr[0])

    def __getitem__(self, key):
        return self.ptr[0].__getitem__(key)

    def __iter__(self):
        for i in range(self.size[0]):
            yield self[i]

# }}}


# {{{ GetInfo support

def _generic_info_to_python(info):
    type_ = _ffi_pystr(info.type)
    value = _ffi.cast(type_, info.value)

    if info.opaque_class != _lib.CLASS_NONE:
        klass = {
            _lib.CLASS_PLATFORM: Platform,
            _lib.CLASS_DEVICE: Device,
            _lib.CLASS_KERNEL: Kernel,
            _lib.CLASS_CONTEXT: Context,
            _lib.CLASS_BUFFER: Buffer,
            _lib.CLASS_PROGRAM: _Program,
            _lib.CLASS_EVENT: Event,
            _lib.CLASS_COMMAND_QUEUE: CommandQueue
            }[info.opaque_class]

        if klass is _Program:
            def create_inst(val):
                from pyopencl import Program
                return Program(_Program._create(val))

        else:
            create_inst = klass._create

        if type_.endswith(']'):
            ret = list(map(create_inst, value))
            _lib.free_pointer(info.value)
            return ret
        else:
            return create_inst(value)

    if type_ == 'char*':
        ret = _ffi_pystr(value)
    elif type_ == 'cl_device_topology_amd*':
        ret = DeviceTopologyAmd(
                value.pcie.bus, value.pcie.device, value.pcie.function)
    elif type_.startswith('char*['):
        ret = list(map(_ffi_pystr, value))
        _lib.free_pointer_array(info.value, len(value))
    elif type_.endswith(']'):
        if type_.startswith('char['):
            # This is usually a CL binary, which may contain NUL characters
            # that should be preserved.
            ret = _bytes(_ffi.buffer(value))

        elif type_.startswith('generic_info['):
            ret = list(map(_generic_info_to_python, value))
        elif type_.startswith('cl_image_format['):
            ret = [ImageFormat(imf.image_channel_order,
                               imf.image_channel_data_type)
                   for imf in value]
        else:
            ret = list(value)
    else:
        ret = value[0]
    if info.dontfree == 0:
        _lib.free_pointer(info.value)
    return ret

# }}}


def _clobj_list(objs):
    if objs is None:
        return _ffi.NULL, 0
    return [ev.ptr for ev in objs], len(objs)


# {{{ common base class

class _Common(object):
    @classmethod
    def _create(cls, ptr):
        self = cls.__new__(cls)
        self.ptr = ptr
        return self
    ptr = _ffi.NULL

    def __del__(self):
        _lib.clobj__delete(self.ptr)

    def __eq__(self, other):
        return other.int_ptr == self.int_ptr

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return _lib.clobj__int_ptr(self.ptr)

    def get_info(self, param):
        info = _ffi.new('generic_info*')
        _handle_error(_lib.clobj__get_info(self.ptr, param, info))
        return _generic_info_to_python(info)

    @property
    def int_ptr(self):
        return _lib.clobj__int_ptr(self.ptr)

    @classmethod
    def from_int_ptr(cls, int_ptr_value, retain=True):
        """Constructs a :mod:`pyopencl` handle from a C-level pointer (given as
        the integer *int_ptr_value*). If *retain* is *True* (the defauult)
        :mod:`pyopencl` will call ``clRetainXXX`` on the provided object. If
        the previous owner of the object will *not* release the reference,
        *retain* should be set to *False*, to effectively transfer ownership to
        :mod:`pyopencl`.

        .. versionchanged:: 2016.1

            *retain* added
        """
        ptr = _ffi.new('clobj_t*')
        _handle_error(_lib.clobj__from_int_ptr(
            ptr, int_ptr_value, getattr(_lib, 'CLASS_%s' % cls._id.upper()),
            retain))
        return cls._create(ptr[0])

# }}}

# }}}


def get_cl_header_version():
    v = _lib.get_cl_version()
    return (v >> (3 * 4),
            (v >> (1 * 4)) & 0xff)


# {{{ constants

_constants = {}


# {{{ constant classes

class _ConstantsNamespace(object):
    def __init__(self):
        raise RuntimeError("This class cannot be instantiated.")

    @classmethod
    def to_string(cls, value, default_format=None):
        for name in dir(cls):
            if (not name.startswith("_") and getattr(cls, name) == value):
                return name

        if default_format is None:
            raise ValueError("a name for value %d was not found in %s"
                    % (value, cls.__name__))
        else:
            return default_format % value


# /!\ If you add anything here, add it to pyopencl/__init__.py as well.

class program_kind(_ConstantsNamespace):  # noqa
    pass


class status_code(_ConstantsNamespace):  # noqa
    pass


class platform_info(_ConstantsNamespace):  # noqa
    pass


class device_type(_ConstantsNamespace):  # noqa
    pass


class device_info(_ConstantsNamespace):  # noqa
    pass


class device_fp_config(_ConstantsNamespace):  # noqa
    pass


class device_mem_cache_type(_ConstantsNamespace):  # noqa
    pass


class device_local_mem_type(_ConstantsNamespace):  # noqa
    pass


class device_exec_capabilities(_ConstantsNamespace):  # noqa
    pass


class device_svm_capabilities(_ConstantsNamespace):  # noqa
    pass


class command_queue_properties(_ConstantsNamespace):  # noqa
    pass


class context_info(_ConstantsNamespace):  # noqa
    pass


class gl_context_info(_ConstantsNamespace):  # noqa
    pass


class context_properties(_ConstantsNamespace):  # noqa
    pass


class command_queue_info(_ConstantsNamespace):  # noqa
    pass


class queue_properties(_ConstantsNamespace):  # noqa
    pass


class mem_flags(_ConstantsNamespace):  # noqa
    @classmethod
    def _writable(cls, flags):
        return flags & (cls.READ_WRITE | cls.WRITE_ONLY)

    @classmethod
    def _hold_host(cls, flags):
        return flags & cls.USE_HOST_PTR

    @classmethod
    def _use_host(cls, flags):
        return flags & (cls.USE_HOST_PTR | cls.COPY_HOST_PTR)

    @classmethod
    def _host_writable(cls, flags):
        return cls._writable(flags) and cls._hold_host(flags)


class svm_mem_flags(_ConstantsNamespace):  # noqa
    pass


class channel_order(_ConstantsNamespace):  # noqa
    pass


class channel_type(_ConstantsNamespace):  # noqa
    pass


class mem_object_type(_ConstantsNamespace):  # noqa
    pass


class mem_info(_ConstantsNamespace):  # noqa
    pass


class image_info(_ConstantsNamespace):  # noqa
    pass


class addressing_mode(_ConstantsNamespace):  # noqa
    pass


class filter_mode(_ConstantsNamespace):  # noqa
    pass


class sampler_info(_ConstantsNamespace):  # noqa
    pass


class map_flags(_ConstantsNamespace):  # noqa
    pass


class program_info(_ConstantsNamespace):  # noqa
    pass


class program_build_info(_ConstantsNamespace):  # noqa
    pass


class program_binary_type(_ConstantsNamespace):  # noqa
    pass


class kernel_info(_ConstantsNamespace):  # noqa
    pass


class kernel_arg_info(_ConstantsNamespace):  # noqa
    pass


class kernel_arg_address_qualifier(_ConstantsNamespace):  # noqa
    pass


class kernel_arg_access_qualifier(_ConstantsNamespace):  # noqa
    pass


class kernel_arg_type_qualifier(_ConstantsNamespace):  # noqa
    pass


class kernel_work_group_info(_ConstantsNamespace):  # noqa
    pass


class event_info(_ConstantsNamespace):  # noqa
    pass


class command_type(_ConstantsNamespace):  # noqa
    pass


class command_execution_status(_ConstantsNamespace):  # noqa
    pass


class profiling_info(_ConstantsNamespace):  # noqa
    pass


class mem_migration_flags(_ConstantsNamespace):  # noqa
    pass


class mem_migration_flags_ext(_ConstantsNamespace):  # noqa
    pass


class device_partition_property(_ConstantsNamespace):  # noqa
    pass


class device_affinity_domain(_ConstantsNamespace):  # noqa
    pass


class gl_object_type(_ConstantsNamespace):  # noqa
    pass


class gl_texture_info(_ConstantsNamespace):  # noqa
    pass


class migrate_mem_object_flags_ext(_ConstantsNamespace):  # noqa
    pass

# }}}

_locals = locals()


# TODO: constant values are cl_ulong
@_ffi.callback('void (*)(const char*, const char* name, int64_t value)')
def _constant_callback(type_, name, value):
    setattr(_locals[_ffi_pystr(type_)], _ffi_pystr(name), value)  # noqa


_lib.populate_constants(_constant_callback)

del _locals
del _constant_callback

# }}}


# {{{ exceptions

class Error(Exception):
    class _ErrorRecord(object):
        __slots__ = ('_routine', '_code', '_what')

        def __init__(self, msg='', code=0, routine=''):
            self._routine = routine
            assert isinstance(code, six.integer_types)
            self._code = code
            self._what = msg

        def routine(self):
            return self._routine

        def code(self):
            return self._code

        def what(self):
            return self._what

    def __init__(self, *a, **kw):
        if len(a) == 1 and not kw and hasattr(a[0], 'what'):
            super(Error, self).__init__(a[0])
        else:
            super(Error, self).__init__(self._ErrorRecord(*a, **kw))

    def __str__(self):
        val = self.args[0]
        try:
            val.routine
        except AttributeError:
            return str(val)
        else:
            result = ""
            if val.code() != status_code.SUCCESS:
                result = status_code.to_string(
                        val.code(), "<unknown error %d>")
            routine = val.routine()
            if routine:
                result = "%s failed: %s" % (routine, result)
            what = val.what()
            if what:
                if result:
                    result += " - "
                result += what
            return result

    @property
    def code(self):
        return self.args[0].code()

    @property
    def routine(self):
        return self.args[0].routine()

    @property
    def what(self):
        return self.args[0].what()

    def is_out_of_memory(self):
        # matches C implementation in src/c_wrapper/error.h
        val = self.args[0]

        return (val.code == status_code.MEM_OBJECT_ALLOCATION_FAILURE
                or val.code == status_code.OUT_OF_RESOURCES
                or val.code == status_code.OUT_OF_HOST_MEMORY)


class MemoryError(Error):
    pass


class LogicError(Error):
    pass


_py_RuntimeError = RuntimeError


class RuntimeError(Error):
    pass


def _handle_error(error):
    if error == _ffi.NULL:
        return
    if error.other == 1:
        # non-pyopencl exceptions are handled here
        e = _py_RuntimeError(_ffi_pystr(error.msg))
        _lib.free_pointer(error.msg)
        _lib.free_pointer(error)
        raise e
    if error.code == status_code.MEM_OBJECT_ALLOCATION_FAILURE:
        klass = MemoryError
    elif error.code <= status_code.INVALID_VALUE:
        klass = LogicError
    elif status_code.INVALID_VALUE < error.code < status_code.SUCCESS:
        klass = RuntimeError
    else:
        klass = Error

    e = klass(routine=_ffi_pystr(error.routine),
              code=error.code, msg=_ffi_pystr(error.msg))
    _lib.free_pointer(error.routine)
    _lib.free_pointer(error.msg)
    _lib.free_pointer(error)
    raise e

# }}}


# {{{ Platform

class Platform(_Common):
    _id = 'platform'

    def get_devices(self, device_type=device_type.ALL):
        devices = _CArray(_ffi.new('clobj_t**'))
        _handle_error(_lib.platform__get_devices(
            self.ptr, devices.ptr, devices.size, device_type))
        return [Device._create(devices.ptr[0][i])
                for i in range(devices.size[0])]

    def __repr__(self):
        return "<pyopencl.Platform '%s' at 0x%x>" % (self.name, self.int_ptr)

    def _get_cl_version(self):
        import re
        version_string = self.version
        match = re.match(r"^OpenCL ([0-9]+)\.([0-9]+) .*$", version_string)
        if match is None:
            raise RuntimeError("platform %s returned non-conformant "
                               "platform version string '%s'" %
                               (self, version_string))

        return int(match.group(1)), int(match.group(2))


def unload_platform_compiler(plat):
    _handle_error(_lib.platform__unload_compiler(plat.ptr))


def get_platforms():
    platforms = _CArray(_ffi.new('clobj_t**'))
    _handle_error(_lib.get_platforms(platforms.ptr, platforms.size))
    return [Platform._create(platforms.ptr[0][i])
            for i in range(platforms.size[0])]

# }}}


# {{{ Device

class Device(_Common):
    _id = 'device'

    def create_sub_devices(self, props):
        props = tuple(props) + (0,)
        devices = _CArray(_ffi.new('clobj_t**'))
        _handle_error(_lib.device__create_sub_devices(
            self.ptr, devices.ptr, devices.size, props))
        return [Device._create(devices.ptr[0][i])
                for i in range(devices.size[0])]

    def __repr__(self):
        return "<pyopencl.Device '%s' on '%s' at 0x%x>" % (
                self.name.strip(), self.platform.name.strip(), self.int_ptr)

    @property
    def persistent_unique_id(self):
        return (self.vendor, self.vendor_id, self.name, self.version)

# }}}


# {{{ Context

def _parse_context_properties(properties):
    if properties is None:
        return _ffi.NULL

    props = []
    for prop_tuple in properties:
        if len(prop_tuple) != 2:
            raise RuntimeError("property tuple must have length 2",
                               status_code.INVALID_VALUE, "Context")

        prop, value = prop_tuple
        if prop is None:
            raise RuntimeError("invalid context property",
                               status_code.INVALID_VALUE, "Context")

        props.append(prop)
        if prop == context_properties.PLATFORM:
            props.append(value.int_ptr)

        elif prop == getattr(context_properties, "WGL_HDC_KHR", None):
            props.append(value)

        elif prop in [getattr(context_properties, key, None) for key in (
                'CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE',
                'GL_CONTEXT_KHR',
                'EGL_DISPLAY_KHR',
                'GLX_DISPLAY_KHR',
                'CGL_SHAREGROUP_KHR',
                )]:

            from ctypes import _Pointer, addressof
            if isinstance(value, _Pointer):
                val = addressof(value)
            else:
                val = int(value)

            if not val:
                raise LogicError("You most likely have not initialized "
                                 "OpenGL properly.",
                                 status_code.INVALID_VALUE, "Context")
            props.append(val)
        else:
            raise RuntimeError("invalid context property",
                               status_code.INVALID_VALUE, "Context")
    props.append(0)
    return props


class Context(_Common):
    _id = 'context'

    def __init__(self, devices=None, properties=None, dev_type=None, cache_dir=None):
        c_props = _parse_context_properties(properties)
        status_code = _ffi.new('cl_int*')

        _ctx = _ffi.new('clobj_t*')
        if devices is not None:
            # from device list
            if dev_type is not None:
                raise RuntimeError("one of 'devices' or 'dev_type' "
                                   "must be None",
                                   status_code.INVALID_VALUE, "Context")
            _devices, num_devices = _clobj_list(devices)
            # TODO parameter order? (for clobj_list)
            _handle_error(_lib.create_context(_ctx, c_props,
                                              num_devices, _devices))

        else:
            # from device type
            if dev_type is None:
                dev_type = device_type.DEFAULT
            _handle_error(_lib.create_context_from_type(_ctx, c_props,
                                                        dev_type))

        self.ptr = _ctx[0]
        self.cache_dir = cache_dir

    def __repr__(self):
        return "<pyopencl.Context at 0x%x on %s>" % (self.int_ptr,
                ", ".join(repr(dev) for dev in self.devices))

    @memoize_method
    def _get_cl_version(self):
        return self.devices[0].platform._get_cl_version()

# }}}


# {{{ CommandQueue

class CommandQueue(_Common):
    _id = 'command_queue'

    def __init__(self, context, device=None, properties=None):
        if properties is None:
            properties = 0

        ptr_command_queue = _ffi.new('clobj_t*')

        _handle_error(_lib.create_command_queue(
            ptr_command_queue, context.ptr,
            _ffi.NULL if device is None else device.ptr, properties))

        self.ptr = ptr_command_queue[0]

    def finish(self):
        _handle_error(_lib.command_queue__finish(self.ptr))

    def flush(self):
        _handle_error(_lib.command_queue__flush(self.ptr))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()

    def _get_cl_version(self):
        return self.context._get_cl_version()


# }}}


# {{{ _norm_shape_dtype and cffi_array

def _norm_shape_dtype(shape, dtype, order="C", strides=None, name=""):
    dtype = np.dtype(dtype)
    if not isinstance(shape, tuple):
        try:
            shape = tuple(shape)
        except:
            shape = (shape,)
    if strides is None:
        if order in "cC":
            strides = c_contiguous_strides(dtype.itemsize, shape)
        elif order in "fF":
            strides = f_contiguous_strides(dtype.itemsize, shape)
        else:
            raise RuntimeError("unrecognized order specifier %s" % order,
                               status_code.INVALID_VALUE, name)
    return dtype, shape, strides


class cffi_array(np.ndarray):  # noqa
    __array_priority__ = -100.0

    def __new__(cls, buf, shape, dtype, strides, base=None):
        self = np.ndarray.__new__(cls, shape, dtype=dtype,
                                  buffer=buf, strides=strides)
        if base is None:
            base = buf
        self.__base = base
        return self

    @property
    def base(self):
        return self.__base

# }}}


# {{{ MemoryObjectHolder base class

class MemoryObjectHolder(_Common, _CLKernelArg):
    def get_host_array(self, shape, dtype, order="C"):
        dtype, shape, strides = _norm_shape_dtype(
            shape, dtype, order, None, 'MemoryObjectHolder.get_host_array')
        _hostptr = _ffi.new('void**')
        _size = _ffi.new('size_t*')
        _handle_error(_lib.memory_object__get_host_array(self.ptr, _hostptr,
                                                         _size))
        ary = cffi_array(_ffi.buffer(_hostptr[0], _size[0]), shape,
                         dtype, strides, self)
        if ary.nbytes > _size[0]:
            raise LogicError("Resulting array is larger than memory object.",
                             status_code.INVALID_VALUE,
                             "MemoryObjectHolder.get_host_array")
        return ary

# }}}


# {{{ MemoryObject

class MemoryObject(MemoryObjectHolder):
    def __init__(self, hostbuf=None):
        self.__hostbuf = hostbuf

    def _handle_buf_flags(self, flags):
        if self.__hostbuf is None:
            return _ffi.NULL, 0, None
        if not mem_flags._use_host(flags):
            warnings.warn("'hostbuf' was passed, but no memory flags "
                          "to make use of it.")

        need_retain = mem_flags._hold_host(flags)
        c_hostbuf, hostbuf_size, retained_buf = _c_buffer_from_obj(
            self.__hostbuf, writable=mem_flags._host_writable(flags),
            retain=need_retain)
        if need_retain:
            self.__retained_buf = retained_buf
        return c_hostbuf, hostbuf_size, retained_buf

    @property
    def hostbuf(self):
        return self.__hostbuf

    def release(self):
        _handle_error(_lib.memory_object__release(self.ptr))

# }}}


# {{{ MemoryMap

class MemoryMap(_Common):
    """"
    .. automethod:: release

    This class may also be used as a context manager in a ``with`` statement.
    """

    @classmethod
    def _create(cls, ptr, shape, typestr, strides):
        self = _Common._create.__func__(cls, ptr)
        self.__array_interface__ = {
            'shape': shape,
            'typestr': typestr,
            'strides': strides,
            'data': (int(_lib.clobj__int_ptr(self.ptr)), False),
            'version': 3
        }
        return self

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self, queue=None, wait_for=None):
        c_wait_for, num_wait_for = _clobj_list(wait_for)
        _event = _ffi.new('clobj_t*')
        _handle_error(_lib.memory_map__release(
            self.ptr, queue.ptr if queue is not None else _ffi.NULL,
            c_wait_for, num_wait_for, _event))
        return Event._create(_event[0])

# }}}


# {{{ _c_buffer_from_obj

if _PYPY:
    # Convert a Python object to a tuple (ptr, num_bytes, ref) to be able to
    # pass a data stream to a C function where @ptr can be passed to a pointer
    # argument and @num_bytes is the number of bytes. For certain types or
    # when @writable or @retain is True, @ref is the object which keep the
    # pointer converted from @ptr object valid.

    def _c_buffer_from_obj(obj, writable=False, retain=False):
        if isinstance(obj, bytes):
            if writable:
                # bytes is not writable
                raise TypeError('expected an object with a writable '
                                'buffer interface.')
            if retain:
                buf = _ffi.new('char[]', obj)
                return (buf, len(obj), buf)
            return (obj, len(obj), obj)
        elif isinstance(obj, np.ndarray):
            # numpy array
            return (_ffi.cast('void*', obj.__array_interface__['data'][0]),
                    obj.nbytes, obj)
        elif isinstance(obj, np.generic):
            if writable or retain:
                raise TypeError('expected an object with a writable '
                                'buffer interface.')

            return (_ffi.cast('void*', memoryview(obj)._pypy_raw_address()),
                    obj.itemsize, obj)
        else:
            raise LogicError("PyOpencl on PyPy only accepts numpy arrays "
                             "and scalars arguments", status_code.INVALID_VALUE)

elif sys.version_info >= (2, 7, 4):
    import ctypes
    try:
        # Python 2.6 doesn't have this.
        _ssize_t = ctypes.c_ssize_t
    except AttributeError:
        _ssize_t = ctypes.c_size_t

    def _c_buffer_from_obj(obj, writable=False, retain=False):
        # {{{ try the numpy array interface first

        # avoid slow ctypes-based buffer interface wrapper

        ary_intf = getattr(obj, "__array_interface__", None)
        if ary_intf is not None:
            buf_base, is_read_only = ary_intf["data"]
            return (
                    _ffi.cast('void*', buf_base + ary_intf.get("offset", 0)),
                    obj.nbytes,
                    obj)

        # }}}

        # {{{ fall back to the old CPython buffer protocol API

        from pyopencl._buffers import Py_buffer, PyBUF_ANY_CONTIGUOUS, PyBUF_WRITABLE

        flags = PyBUF_ANY_CONTIGUOUS
        if writable:
            flags |= PyBUF_WRITABLE

        with Py_buffer.from_object(obj, flags) as buf:
            return _ffi.cast('void*', buf.buf), buf.len, obj

        # }}}

else:
    # Py2.6 and below

    import ctypes
    try:
        # Python 2.6 doesn't have this.
        _ssize_t = ctypes.c_ssize_t
    except AttributeError:
        _ssize_t = ctypes.c_size_t

    def _c_buffer_from_obj(obj, writable=False, retain=False):
        # {{{ fall back to the old CPython buffer protocol API

        addr = ctypes.c_void_p()
        length = _ssize_t()

        try:
            if writable:
                ctypes.pythonapi.PyObject_AsWriteBuffer(
                    ctypes.py_object(obj), ctypes.byref(addr),
                    ctypes.byref(length))
            else:
                ctypes.pythonapi.PyObject_AsReadBuffer(
                    ctypes.py_object(obj), ctypes.byref(addr),
                    ctypes.byref(length))

                # ctypes check exit status of these, so no need to check
                # for errors.
        except TypeError:
            raise LogicError(routine=None, code=status_code.INVALID_VALUE,
                             msg=("un-sized (pure-Python) types not "
                                  "acceptable as arguments"))
        # }}}

        return _ffi.cast('void*', addr.value), length.value, obj

# }}}


# {{{ Buffer

class Buffer(MemoryObject):
    _id = 'buffer'

    def __init__(self, context, flags, size=0, hostbuf=None):
        MemoryObject.__init__(self, hostbuf)
        c_hostbuf, hostbuf_size, retained_buf = self._handle_buf_flags(flags)
        if hostbuf is not None:
            if size > hostbuf_size:
                raise RuntimeError("Specified size is greater than host "
                                   "buffer size",
                                   status_code.INVALID_VALUE, "Buffer")
            if size == 0:
                size = hostbuf_size

        ptr_buffer = _ffi.new('clobj_t*')
        _handle_error(_lib.create_buffer(
            ptr_buffer, context.ptr, flags, size, c_hostbuf))
        self.ptr = ptr_buffer[0]

    def get_sub_region(self, origin, size, flags=0):
        _sub_buf = _ffi.new('clobj_t*')
        _handle_error(_lib.buffer__get_sub_region(_sub_buf, self.ptr, origin,
                                                  size, flags))
        sub_buf = self._create(_sub_buf[0])
        MemoryObject.__init__(sub_buf, None)
        return sub_buf

    def __getitem__(self, idx):
        if not isinstance(idx, slice):
            raise TypeError("buffer subscript must be a slice object")

        start, stop, stride = idx.indices(self.size)
        if stride != 1:
            raise ValueError("Buffer slice must have stride 1",
                               status_code.INVALID_VALUE, "Buffer.__getitem__")

        assert start <= stop

        size = stop - start
        return self.get_sub_region(start, size)

# }}}


# {{{ SVMAllocation

class SVMAllocation(object):
    """An object whose lifetime is tied to an allocation of shared virtual memory.

    .. note::

        Most likely, you will not want to use this directly, but rather
        :func:`svm_empty` and related functions which allow access to this
        functionality using a friendlier, more Pythonic interface.

    .. versionadded:: 2016.2

    .. automethod:: __init__(self, ctx, size, alignment, flags=None)
    .. automethod:: release
    .. automethod:: enqueue_release
    """
    def __init__(self, ctx, size, alignment, flags, _interface=None):
        """
        :arg ctx: a :class:`Context`
        :arg flags: some of :class:`svm_mem_flags`.
        """

        self.ptr = None

        ptr = _ffi.new('void**')
        _handle_error(_lib.svm_alloc(
            ctx.ptr, flags, size, alignment,
            ptr))

        self.ctx = ctx
        self.ptr = ptr[0]
        self.is_fine_grain = flags & svm_mem_flags.SVM_FINE_GRAIN_BUFFER

        if _interface is not None:
            read_write = (
                    flags & mem_flags.WRITE_ONLY != 0
                    or flags & mem_flags.READ_WRITE != 0)
            _interface["data"] = (
                    int(_ffi.cast("intptr_t", self.ptr)), not read_write)
            self.__array_interface__ = _interface

    def __del__(self):
        if self.ptr is not None:
            self.release()

    def release(self):
        _handle_error(_lib.svm_free(self.ctx.ptr, self.ptr))
        self.ptr = None

    def enqueue_release(self, queue, wait_for=None):
        """
        :arg flags: a combination of :class:`pyopencl.map_flags`
        :returns: a :class:`pyopencl.Event`

        |std-enqueue-blurb|
        """
        ptr_event = _ffi.new('clobj_t*')
        c_wait_for, num_wait_for = _clobj_list(wait_for)
        _handle_error(_lib.enqueue_svm_free(
            ptr_event, queue.ptr, 1, self.ptr,
            c_wait_for, num_wait_for))

        self.ctx = None
        self.ptr = None

        return Event._create(ptr_event[0])

# }}}


# {{{ SVM

# TODO add clSetKernelExecInfo

class SVM(_CLKernelArg):
    """Tags an object exhibiting the Python buffer interface (such as a
    :class:`numpy.ndarray`) as referring to shared virtual memory.

    Depending on the features of the OpenCL implementation, the following
    types of objects may be passed to/wrapped in this type:

    *   coarse-grain shared memory as returned by (e.g.) :func:`csvm_empty`
        for any implementation of OpenCL 2.0.

        This is how coarse-grain SVM may be used from both host and device::

            svm_ary = cl.SVM(cl.csvm_empty(ctx, 1000, np.float32, alignment=64))
            assert isinstance(svm_ary.mem, np.ndarray)

            with svm_ary.map_rw(queue) as ary:
                ary.fill(17)  # use from host

            prg.twice(queue, svm_ary.mem.shape, None, svm_ary)

    *   fine-grain shared memory as returned by (e.g.) :func:`fsvm_empty`,
        if the implementation supports fine-grained shared virtual memory.
        This memory may directly be passed to a kernel::

            ary = cl.fsvm_empty(ctx, 1000, np.float32)
            assert isinstance(ary, np.ndarray)

            prg.twice(queue, ary.shape, None, cl.SVM(ary))
            queue.finish() # synchronize
            print(ary) # access from host

        Observe how mapping (as needed in coarse-grain SVM) is no longer
        necessary.

    *   any :class:`numpy.ndarray` (or other Python object with a buffer
        interface) if the implementation supports fine-grained *system* shared
        virtual memory.

        This is how plain :mod:`numpy` arrays may directly be passed to a
        kernel::

            ary = np.zeros(1000, np.float32)
            prg.twice(queue, ary.shape, None, cl.SVM(ary))
            queue.finish() # synchronize
            print(ary) # access from host

    Objects of this type may be passed to kernel calls and :func:`enqueue_copy`.
    Coarse-grain shared-memory *must* be mapped into host address space using
    :meth:`map` before being accessed through the :mod:`numpy` interface.

    .. note::

        This object merely serves as a 'tag' that changes the behavior
        of functions to which it is passed. It has no special management
        relationship to the memory it tags. For example, it is permissible
        to grab a :mod:`numpy.array` out of :attr:`SVM.mem` of one
        :class:`SVM` instance and use the array to construct another.
        Neither of the tags need to be kept alive.

    .. versionadded:: 2016.2

    .. attribute:: mem

        The wrapped object.

    .. automethod:: __init__
    .. automethod:: map
    .. automethod:: map_ro
    .. automethod:: map_rw
    .. automethod:: as_buffer
    """

    def __init__(self, mem):
        self.mem = mem

    def map(self, queue, flags, is_blocking=True, wait_for=None):
        """
        :arg is_blocking: If *False*, subsequent code must wait on
            :attr:`SVMMap.event` in the returned object before accessing the
            mapped memory.
        :arg flags: a combination of :class:`pyopencl.map_flags`, defaults to
            read-write.
        :returns: an :class:`SVMMap` instance

        |std-enqueue-blurb|
        """
        writable = bool(
            flags & (map_flags.WRITE | map_flags.WRITE_INVALIDATE_REGION))
        c_buf, size, _ = _c_buffer_from_obj(self.mem, writable=writable)

        ptr_event = _ffi.new('clobj_t*')
        c_wait_for, num_wait_for = _clobj_list(wait_for)
        _handle_error(_lib.enqueue_svm_map(
            ptr_event, queue.ptr, is_blocking, flags,
            c_buf, size,
            c_wait_for, num_wait_for))

        evt = Event._create(ptr_event[0])
        return SVMMap(self, queue, evt)

    def map_ro(self, queue, is_blocking=True, wait_for=None):
        """Like :meth:`map`, but with *flags* set for a read-only map."""

        return self.map(queue, map_flags.READ,
                is_blocking=is_blocking, wait_for=wait_for)

    def map_rw(self, queue, is_blocking=True, wait_for=None):
        """Like :meth:`map`, but with *flags* set for a read-only map."""

        return self.map(queue, map_flags.READ | map_flags.WRITE,
                is_blocking=is_blocking, wait_for=wait_for)

    def _enqueue_unmap(self, queue, wait_for=None):
        c_buf, _, _ = _c_buffer_from_obj(self.mem)

        ptr_event = _ffi.new('clobj_t*')
        c_wait_for, num_wait_for = _clobj_list(wait_for)
        _handle_error(_lib.enqueue_svm_unmap(
            ptr_event, queue.ptr,
            c_buf,
            c_wait_for, num_wait_for))

        return Event._create(ptr_event[0])

    def as_buffer(self, ctx, flags=None):
        """
        :arg ctx: a :class:`Context`
        :arg flags: a combination of :class:`pyopencl.map_flags`, defaults to
            read-write.
        :returns: a :class:`Buffer` corresponding to *self*.

        The memory referred to by this object must not be freed before
        the returned :class:`Buffer` is released.
        """

        if flags is None:
            flags = mem_flags.READ_WRITE

        return Buffer(ctx, flags, size=self.mem.nbytes, hostbuf=self.mem)


def _enqueue_svm_memcpy(queue, dst, src, size=None,
        wait_for=None, is_blocking=True):
    dst_buf, dst_size, _ = _c_buffer_from_obj(dst, writable=True)
    src_buf, src_size, _ = _c_buffer_from_obj(src, writable=False)

    if size is None:
        size = min(dst_size, src_size)

    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_svm_memcpy(
        ptr_event, queue.ptr,  bool(is_blocking),
        dst_buf, src_buf, size,
        c_wait_for, num_wait_for,
        NannyEvent._handle((dst_buf, src_buf))))

    return NannyEvent._create(ptr_event[0])


def enqueue_svm_memfill(queue, dest, pattern, byte_count=None, wait_for=None):
    """Fill shared virtual memory with a pattern.

    :arg dest: a Python buffer object, optionally wrapped in an :class:`SVM` object
    :arg pattern: a Python buffer object (e.g. a :class:`numpy.ndarray` with the
        fill pattern to be used.
    :arg byte_count: The size of the memory to be fill. Defaults to the
        entirety of *dest*.

    |std-enqueue-blurb|

    .. versionadded:: 2016.2
    """

    if isinstance(dest, SVM):
        dest = dest.mem

    dst_buf, dst_size, _ = _c_buffer_from_obj(dest, writable=True)
    pattern_buf, pattern_size, _ = _c_buffer_from_obj(pattern, writable=False)

    if byte_count is None:
        byte_count = dst_size

    # pattern is copied, no need to nanny.
    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_svm_memfill(
        ptr_event, queue.ptr,
        dst_buf, pattern_buf, pattern_size, byte_count,
        c_wait_for, num_wait_for))

    return Event._create(ptr_event[0])


def enqueue_svm_migratemem(queue, svms, flags, wait_for=None):
    """
    :arg svms: a collection of Python buffer objects (e.g. :mod:`numpy`
        arrrays), optionally wrapped in :class:`SVM` objects.
    :arg flags: a combination of :class:`mem_migration_flags`

    |std-enqueue-blurb|

    .. versionadded:: 2016.2

    This function requires OpenCL 2.1.
    """

    svm_pointers = _ffi.new('void *', len(svms))
    sizes = _ffi.new('size_t', len(svms))

    for i, svm in enumerate(svms):
        if isinstance(svm, SVM):
            svm = svm.mem

        buf, size, _ = _c_buffer_from_obj(svm, writable=False)
        svm_pointers[i] = buf
        sizes[i] = size

    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_svm_memfill(
        ptr_event, queue.ptr,
        len(svms), svm_pointers, sizes, flags,
        c_wait_for, num_wait_for))

    return Event._create(ptr_event[0])

# }}}


# {{{ SVMMap

class SVMMap(_CLKernelArg):
    """
    .. attribute:: event

    .. versionadded:: 2016.2

    .. automethod:: release

    This class may also be used as a context manager in a ``with`` statement.
    :meth:`release` will be called upon exit from the ``with`` region.
    The value returned to the ``as`` part of the context manager is the
    mapped Python object (e.g. a :mod:`numpy` array).
    """
    def __init__(self, svm, queue, event):
        self.svm = svm
        self.queue = queue
        self.event = event

    def __del__(self):
        if self.svm is not None:
            self.release()

    def __enter__(self):
        return self.svm.mem

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()

    def release(self, queue=None, wait_for=None):
        """
        :arg queue: a :class:`pyopencl.CommandQueue`. Defaults to the one
            with which the map was created, if not specified.
        :returns: a :class:`pyopencl.Event`

        |std-enqueue-blurb|
        """

        evt = self.svm._enqueue_unmap(self.queue)
        self.svm = None

        return evt

# }}}


# {{{ Program

class CompilerWarning(UserWarning):
    pass


def compiler_output(text):
    import os
    from warnings import warn
    if int(os.environ.get("PYOPENCL_COMPILER_OUTPUT", "0")):
        warn(text, CompilerWarning)
    else:
        warn("Non-empty compiler output encountered. Set the "
                "environment variable PYOPENCL_COMPILER_OUTPUT=1 "
                "to see more.", CompilerWarning)


class _Program(_Common):
    _id = 'program'

    def __init__(self, *args):
        if len(args) == 2:
            ctx, source = args
            from pyopencl.tools import is_spirv
            if is_spirv(source):
                self._init_il(ctx, source)
            else:
                self._init_source(ctx, source)
        else:
            self._init_binary(*args)

    def _init_source(self, context, src):
        ptr_program = _ffi.new('clobj_t*')
        _handle_error(_lib.create_program_with_source(
            ptr_program, context.ptr, _to_cstring(src)))
        self.ptr = ptr_program[0]

    def _init_il(self, context, il):
        ptr_program = _ffi.new('clobj_t*')
        _handle_error(_lib.create_program_with_il(
            ptr_program, context.ptr, il, len(il)))
        self.ptr = ptr_program[0]

    def _init_binary(self, context, devices, binaries):
        if len(devices) != len(binaries):
            raise RuntimeError("device and binary counts don't match",
                               status_code.INVALID_VALUE,
                               "create_program_with_binary")

        ptr_program = _ffi.new('clobj_t*')
        ptr_devices, num_devices = _clobj_list(devices)
        ptr_binaries = [_ffi.new('unsigned char[]', binary)
                        for binary in binaries]
        binary_sizes = [len(b) for b in binaries]

        # TODO parameter order? (for clobj_list)
        _handle_error(_lib.create_program_with_binary(
            ptr_program, context.ptr, num_devices, ptr_devices,
            ptr_binaries, binary_sizes))

        self.ptr = ptr_program[0]

    def kind(self):
        kind = _ffi.new('int*')
        _handle_error(_lib.program__kind(self.ptr, kind))
        return kind[0]

    def _build(self, options=None, devices=None):
        if options is None:
            options = b""
        # TODO? reverse parameter order
        ptr_devices, num_devices = _clobj_list(devices)
        _handle_error(_lib.program__build(self.ptr, options,
                                          num_devices, ptr_devices))

    def get_build_info(self, device, param):
        info = _ffi.new('generic_info *')
        _handle_error(_lib.program__get_build_info(
            self.ptr, device.ptr, param, info))
        return _generic_info_to_python(info)

    def compile(self, options="", devices=None, headers=[]):
        _devs, num_devs = _clobj_list(devices)
        _prgs, names = list(zip(*((prg.ptr, _to_cstring(name))
                             for (name, prg) in headers)))
        _handle_error(_lib.program__compile(
            self.ptr, _to_cstring(options), _devs, num_devs,
            _prgs, names, len(names)))

    @classmethod
    def link(cls, context, programs, options="", devices=None):
        _devs, num_devs = _clobj_list(devices)
        _prgs, num_prgs = _clobj_list(programs)
        _prg = _ffi.new('clobj_t*')
        _handle_error(_lib.program__link(
            _prg, context.ptr, _prgs, num_prgs, _to_cstring(options),
            _devs, num_devs))
        return cls._create(_prg[0])

    @classmethod
    def create_with_builtin_kernels(cls, context, devices, kernel_names):
        _devs, num_devs = _clobj_list(devices)
        _prg = _ffi.new('clobj_t*')
        _handle_error(_lib.program__create_with_builtin_kernels(
            _prg, context.ptr, _devs, num_devs, _to_cstring(kernel_names)))
        return cls._create(_prg[0])

    def all_kernels(self):
        knls = _CArray(_ffi.new('clobj_t**'))
        _handle_error(_lib.program__all_kernels(
            self.ptr, knls.ptr, knls.size))
        return [
                Kernel
                ._create(knls.ptr[0][i])
                ._setup(self)
                for i in range(knls.size[0])]

    def _get_build_logs(self):
        build_logs = []
        for dev in self.get_info(program_info.DEVICES):
            try:
                log = self.get_build_info(dev, program_build_info.LOG)
            except:
                log = "<error retrieving log>"

            build_logs.append((dev, log))

        return build_logs

    def build(self, options_bytes, devices=None):
        err = None
        try:
            self._build(options=options_bytes, devices=devices)
        except Error as e:
            msg = e.what + "\n\n" + (75*"="+"\n").join(
                    "Build on %s:\n\n%s" % (dev, log)
                    for dev, log in self._get_build_logs())
            code = e.code
            routine = e.routine

            err = RuntimeError(
                    Error._ErrorRecord(
                        msg=msg,
                        code=code,
                        routine=routine))

        if err is not None:
            # Python 3.2 outputs the whole list of currently active exceptions
            # This serves to remove one (redundant) level from that nesting.
            raise err

        message = (75*"="+"\n").join(
                "Build on %s succeeded, but said:\n\n%s" % (dev, log)
                for dev, log in self._get_build_logs()
                if log is not None and log.strip())

        if message:
            if self.kind() == program_kind.SOURCE:
                build_type = "From-source build"
            elif self.kind() == program_kind.BINARY:
                build_type = "From-binary build"
            else:
                build_type = "Build"

            compiler_output("%s succeeded, but resulted in non-empty logs:\n%s"
                    % (build_type, message))

        return self

# }}}


class LocalMemory(_CLKernelArg):
    __slots__ = ('_size',)

    def __init__(self, size):
        self._size = size

    @property
    def size(self):
        return self._size


# {{{ Kernel

# {{{ arg packing helpers

_size_t_char = ({
    8: 'Q',
    4: 'L',
    2: 'H',
    1: 'B',
})[_ffi.sizeof('size_t')]
_type_char_map = {
    'n': _size_t_char.lower(),
    'N': _size_t_char
}
del _size_t_char

# }}}


class Kernel(_Common):
    _id = 'kernel'

    def __init__(self, program, name):
        if not isinstance(program, _Program):
            program = program._get_prg()

        ptr_kernel = _ffi.new('clobj_t*')
        _handle_error(_lib.create_kernel(ptr_kernel, program.ptr,
                                         _to_cstring(name)))
        self.ptr = ptr_kernel[0]

        self._setup(program)

    def _setup(self, prg):
        self._source = getattr(prg, "_source", None)

        self._generate_naive_call()
        self._wg_info_cache = {}
        return self

    # {{{ code generation for __call__, set_args

    def _set_set_args_body(self, body, num_passed_args):
        from pytools.py_codegen import (
                PythonFunctionGenerator,
                PythonCodeGenerator,
                Indentation)

        arg_names = ["arg%d" % i for i in range(num_passed_args)]

        # {{{ wrap in error handler

        err_gen = PythonCodeGenerator()

        def gen_error_handler():
            err_gen("""
                if current_arg is not None:
                    args = [{args}]
                    advice = ""
                    from pyopencl.array import Array
                    if isinstance(args[current_arg], Array):
                        advice = " (perhaps you meant to pass 'array.data' " \
                            "instead of the array itself?)"

                    raise _cl.LogicError(
                            "when processing argument #%d (1-based): %s%s"
                            % (current_arg+1, str(e), advice))
                else:
                    raise
                """
                .format(args=", ".join(arg_names)))
            err_gen("")

        err_gen("try:")
        with Indentation(err_gen):
            err_gen.extend(body)
        err_gen("except TypeError as e:")
        with Indentation(err_gen):
            gen_error_handler()
        err_gen("except _cl.LogicError as e:")
        with Indentation(err_gen):
            gen_error_handler()

        # }}}

        def add_preamble(gen):
            gen.add_to_preamble(
                "import numpy as np")
            gen.add_to_preamble(
                "import pyopencl.cffi_cl as _cl")
            gen.add_to_preamble(
                "from pyopencl.cffi_cl import _lib, "
                "_ffi, _handle_error, _CLKernelArg")
            gen.add_to_preamble("from pyopencl import status_code")
            gen.add_to_preamble("from struct import pack")
            gen.add_to_preamble("")

        # {{{ generate _enqueue

        gen = PythonFunctionGenerator("enqueue_knl_%s" % self.function_name,
                ["self", "queue", "global_size", "local_size"]
                + arg_names
                + ["global_offset=None", "g_times_l=None", "wait_for=None"])

        add_preamble(gen)
        gen.extend(err_gen)

        gen("""
            return _cl.enqueue_nd_range_kernel(queue, self, global_size, local_size,
                    global_offset, wait_for, g_times_l=g_times_l)
            """)

        self._enqueue = gen.get_function()

        # }}}

        # {{{ generate set_args

        gen = PythonFunctionGenerator("_set_args", ["self"] + arg_names)

        add_preamble(gen)
        gen.extend(err_gen)

        self._set_args = gen.get_function()

        # }}}

    def _generate_buffer_arg_setter(self, gen, arg_idx, buf_var):
        from pytools.py_codegen import Indentation

        if _CPY2:
            # https://github.com/numpy/numpy/issues/5381
            gen("if isinstance({buf_var}, np.generic):".format(buf_var=buf_var))
            with Indentation(gen):
                gen("{buf_var} = np.getbuffer({buf_var})".format(buf_var=buf_var))

        gen("""
            c_buf, sz, _ = _cl._c_buffer_from_obj({buf_var})
            status = _lib.kernel__set_arg_buf(self.ptr, {arg_idx}, c_buf, sz)
            if status != _ffi.NULL:
                _handle_error(status)
            """
            .format(arg_idx=arg_idx, buf_var=buf_var))

    def _generate_bytes_arg_setter(self, gen, arg_idx, buf_var):
        gen("""
            status = _lib.kernel__set_arg_buf(self.ptr, {arg_idx},
                {buf_var}, len({buf_var}))
            if status != _ffi.NULL:
                _handle_error(status)
            """
            .format(arg_idx=arg_idx, buf_var=buf_var))

    def _generate_generic_arg_handler(self, gen, arg_idx, arg_var):
        from pytools.py_codegen import Indentation

        gen("""
            if {arg_var} is None:
                status = _lib.kernel__set_arg_null(self.ptr, {arg_idx})
                if status != _ffi.NULL:
                    _handle_error(status)
            elif isinstance({arg_var}, _cl._CLKernelArg):
                self._set_arg_clkernelarg({arg_idx}, {arg_var})
            """
            .format(arg_idx=arg_idx, arg_var=arg_var))

        gen("else:")
        with Indentation(gen):
            self._generate_buffer_arg_setter(gen, arg_idx, arg_var)

    def _generate_naive_call(self):
        num_args = self.num_args

        from pytools.py_codegen import PythonCodeGenerator
        gen = PythonCodeGenerator()

        if num_args == 0:
            gen("pass")

        for i in range(num_args):
            gen("# process argument {arg_idx}".format(arg_idx=i))
            gen("")
            gen("current_arg = {arg_idx}".format(arg_idx=i))
            self._generate_generic_arg_handler(gen, i, "arg%d" % i)
            gen("")

        self._set_set_args_body(gen, num_args)

    def set_scalar_arg_dtypes(self, scalar_arg_dtypes):
        self._scalar_arg_dtypes = scalar_arg_dtypes

        # {{{ arg counting bug handling

        # For example:
        # https://github.com/pocl/pocl/issues/197
        # (but Apple CPU has a similar bug)

        work_around_arg_count_bug = False
        warn_about_arg_count_bug = False

        from pyopencl.characterize import has_struct_arg_count_bug

        count_bug_per_dev = [
                has_struct_arg_count_bug(dev)
                for dev in self.context.devices]

        from pytools import single_valued
        if any(count_bug_per_dev):
            if all(count_bug_per_dev):
                work_around_arg_count_bug = single_valued(count_bug_per_dev)
            else:
                warn_about_arg_count_bug = True

        fp_arg_count = 0

        # }}}

        cl_arg_idx = 0

        from pytools.py_codegen import PythonCodeGenerator
        gen = PythonCodeGenerator()

        if not scalar_arg_dtypes:
            gen("pass")

        for arg_idx, arg_dtype in enumerate(scalar_arg_dtypes):
            gen("# process argument {arg_idx}".format(arg_idx=arg_idx))
            gen("")
            gen("current_arg = {arg_idx}".format(arg_idx=arg_idx))
            arg_var = "arg%d" % arg_idx

            if arg_dtype is None:
                self._generate_generic_arg_handler(gen, cl_arg_idx, arg_var)
                cl_arg_idx += 1
                gen("")
                continue

            arg_dtype = np.dtype(arg_dtype)

            if arg_dtype.char == "V":
                self._generate_generic_arg_handler(gen, cl_arg_idx, arg_var)
                cl_arg_idx += 1

            elif arg_dtype.kind == "c":
                if warn_about_arg_count_bug:
                    warn("{knl_name}: arguments include complex numbers, and "
                            "some (but not all) of the target devices mishandle "
                            "struct kernel arguments (hence the workaround is "
                            "disabled".format(
                                knl_name=self.function_name, stacklevel=2))

                if arg_dtype == np.complex64:
                    arg_char = "f"
                elif arg_dtype == np.complex128:
                    arg_char = "d"
                else:
                    raise TypeError("unexpected complex type: %s" % arg_dtype)

                if (work_around_arg_count_bug == "pocl"
                        and arg_dtype == np.complex128
                        and fp_arg_count + 2 <= 8):
                    gen(
                            "buf = pack('{arg_char}', {arg_var}.real)"
                            .format(arg_char=arg_char, arg_var=arg_var))
                    self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                    cl_arg_idx += 1
                    gen("current_arg = current_arg + 1000")
                    gen(
                            "buf = pack('{arg_char}', {arg_var}.imag)"
                            .format(arg_char=arg_char, arg_var=arg_var))
                    self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                    cl_arg_idx += 1

                elif (work_around_arg_count_bug == "apple"
                        and arg_dtype == np.complex128
                        and fp_arg_count + 2 <= 8):
                    raise NotImplementedError("No work-around to "
                            "Apple's broken structs-as-kernel arg "
                            "handling has been found. "
                            "Cannot pass complex numbers to kernels.")

                else:
                    gen(
                            "buf = pack('{arg_char}{arg_char}', "
                            "{arg_var}.real, {arg_var}.imag)"
                            .format(arg_char=arg_char, arg_var=arg_var))
                    self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                    cl_arg_idx += 1

                fp_arg_count += 2

            elif arg_dtype.char in "IL" and _CPY26:
                # Prevent SystemError: ../Objects/longobject.c:336: bad
                # argument to internal function

                gen(
                        "buf = pack('{arg_char}', long({arg_var}))"
                        .format(arg_char=arg_dtype.char, arg_var=arg_var))
                self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                cl_arg_idx += 1

            else:
                if arg_dtype.kind == "f":
                    fp_arg_count += 1

                arg_char = arg_dtype.char
                arg_char = _type_char_map.get(arg_char, arg_char)
                gen(
                        "buf = pack('{arg_char}', {arg_var})"
                        .format(
                            arg_char=arg_char,
                            arg_var=arg_var))
                self._generate_bytes_arg_setter(gen, cl_arg_idx, "buf")
                cl_arg_idx += 1

            gen("")

        if cl_arg_idx != self.num_args:
            raise TypeError(
                "length of argument list (%d) and "
                "CL-generated number of arguments (%d) do not agree"
                % (cl_arg_idx, self.num_args))

        self._set_set_args_body(gen, len(scalar_arg_dtypes))

    # }}}

    def set_args(self, *args, **kwargs):
        # Need to duplicate the 'self' argument for dynamically generated  method
        return self._set_args(self, *args, **kwargs)

    def __call__(self, queue, global_size, local_size, *args, **kwargs):
        # __call__ can't be overridden directly, so we need this
        # trampoline hack.
        return self._enqueue(self, queue, global_size, local_size, *args, **kwargs)

    def capture_call(self, filename, queue, global_size, local_size,
            *args, **kwargs):
        from pyopencl.capture_call import capture_kernel_call
        capture_kernel_call(self, filename, queue, global_size, local_size,
                *args, **kwargs)

    def _set_arg_clkernelarg(self, arg_index, arg):
        if isinstance(arg, MemoryObjectHolder):
            _handle_error(_lib.kernel__set_arg_mem(self.ptr, arg_index, arg.ptr))
        elif isinstance(arg, SVM):
            c_buf, _, _ = _c_buffer_from_obj(arg.mem)
            _handle_error(_lib.kernel__set_arg_svm_pointer(
                self.ptr, arg_index, c_buf))
        elif isinstance(arg, Sampler):
            _handle_error(_lib.kernel__set_arg_sampler(self.ptr, arg_index,
                                                       arg.ptr))
        elif isinstance(arg, LocalMemory):
            _handle_error(_lib.kernel__set_arg_buf(self.ptr, arg_index,
                                                   _ffi.NULL, arg.size))
        else:
            raise RuntimeError("unexpected _CLKernelArg subclass"
                               "dimensions", status_code.INVALID_VALUE,
                               "clSetKernelArg")

    def set_arg(self, arg_index, arg):
        # If you change this, also change the kernel call generation logic.
        if arg is None:
            _handle_error(_lib.kernel__set_arg_null(self.ptr, arg_index))
        elif isinstance(arg, _CLKernelArg):
            self._set_arg_clkernelarg(arg_index, arg)
        elif _CPY2 and isinstance(arg, np.generic):
            # https://github.com/numpy/numpy/issues/5381
            c_buf, size, _ = _c_buffer_from_obj(np.getbuffer(arg))
            _handle_error(_lib.kernel__set_arg_buf(self.ptr, arg_index,
                                                   c_buf, size))
        else:
            c_buf, size, _ = _c_buffer_from_obj(arg)
            _handle_error(_lib.kernel__set_arg_buf(self.ptr, arg_index,
                                                   c_buf, size))

    def get_work_group_info(self, param, device):
        try:
            return self._wg_info_cache[param, device]
        except KeyError:
            pass

        info = _ffi.new('generic_info*')
        _handle_error(_lib.kernel__get_work_group_info(
            self.ptr, param, device.ptr, info))
        result = _generic_info_to_python(info)

        self._wg_info_cache[param, device] = result
        return result

    def get_arg_info(self, idx, param):
        info = _ffi.new('generic_info*')
        _handle_error(_lib.kernel__get_arg_info(self.ptr, idx, param, info))
        return _generic_info_to_python(info)

# }}}


# {{{ Event

class Event(_Common):
    _id = 'event'

    def __init__(self):
        pass

    def get_profiling_info(self, param):
        info = _ffi.new('generic_info *')
        _handle_error(_lib.event__get_profiling_info(self.ptr, param, info))
        return _generic_info_to_python(info)

    def wait(self):
        _handle_error(_lib.event__wait(self.ptr))

    def set_callback(self, _type, cb):
        def _func(status):
            cb(status)
        _handle_error(_lib.event__set_callback(self.ptr, _type,
                                               _ffi.new_handle(_func)))


class ProfilingInfoGetter:
    def __init__(self, event):
        self.event = event

    def __getattr__(self, name):
        info_cls = profiling_info

        try:
            inf_attr = getattr(info_cls, name.upper())
        except AttributeError:
            raise AttributeError("%s has no attribute '%s'"
                    % (type(self), name))
        else:
            return self.event.get_profiling_info(inf_attr)

Event.profile = property(ProfilingInfoGetter)


def wait_for_events(wait_for):
    _handle_error(_lib.wait_for_events(*_clobj_list(wait_for)))


class NannyEvent(Event):
    class _Data(object):
        __slots__ = ('ward', 'ref')

        def __init__(self, ward, ref):
            self.ward = ward
            self.ref = ref

    @classmethod
    def _handle(cls, ward, ref=None):
        return _ffi.new_handle(cls._Data(ward, ref))

    def get_ward(self):
        _handle = _lib.nanny_event__get_ward(self.ptr)
        if _handle == _ffi.NULL:
            return
        return _ffi.from_handle(_handle).ward


class UserEvent(Event):
    def __init__(self, ctx):
        _evt = _ffi.new('clobj_t*')
        _handle_error(_lib.create_user_event(_evt, ctx.ptr))
        self.ptr = _evt[0]

    def set_status(self, status):
        _handle_error(_lib.user_event__set_status(self.ptr, status))

# }}}


# {{{ enqueue_nd_range_kernel

def enqueue_nd_range_kernel(queue, kernel, global_work_size, local_work_size,
                            global_work_offset=None, wait_for=None,
                            g_times_l=False):

    work_dim = len(global_work_size)

    if local_work_size is not None:
        if g_times_l:
            work_dim = max(work_dim, len(local_work_size))
        elif work_dim != len(local_work_size):
            raise RuntimeError("global/local work sizes have differing "
                               "dimensions", status_code.INVALID_VALUE,
                               "enqueue_nd_range_kernel")

        if len(local_work_size) < work_dim:
            local_work_size = (local_work_size +
                               (1,) * (work_dim - len(local_work_size)))
        if len(global_work_size) < work_dim:
            global_work_size = (global_work_size +
                                (1,) * (work_dim - len(global_work_size)))
        if g_times_l:
            global_work_size = tuple(
                    global_work_size[i] * local_work_size[i]
                    for i in range(work_dim))

    c_global_work_offset = _ffi.NULL
    if global_work_offset is not None:
        if work_dim != len(global_work_offset):
            raise RuntimeError("global work size and offset have differing "
                               "dimensions", status_code.INVALID_VALUE,
                               "enqueue_nd_range_kernel")

        c_global_work_offset = global_work_offset

    if local_work_size is None:
        local_work_size = _ffi.NULL

    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_nd_range_kernel(
        ptr_event, queue.ptr, kernel.ptr, work_dim, c_global_work_offset,
        global_work_size, local_work_size, c_wait_for, num_wait_for))
    return Event._create(ptr_event[0])

# }}}


# {{{ enqueue_task

def enqueue_task(queue, kernel, wait_for=None):
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_task(
        _event, queue.ptr, kernel.ptr, c_wait_for, num_wait_for))
    return Event._create(_event[0])

# }}}


# {{{ _enqueue_marker_*

def _enqueue_marker_with_wait_list(queue, wait_for=None):
    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_marker_with_wait_list(
        ptr_event, queue.ptr, c_wait_for, num_wait_for))
    return Event._create(ptr_event[0])


def _enqueue_marker(queue):
    ptr_event = _ffi.new('clobj_t*')
    _handle_error(_lib.enqueue_marker(ptr_event, queue.ptr))
    return Event._create(ptr_event[0])

# }}}


# {{{ _enqueue_barrier_*

def _enqueue_barrier_with_wait_list(queue, wait_for=None):
    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_barrier_with_wait_list(
        ptr_event, queue.ptr, c_wait_for, num_wait_for))
    return Event._create(ptr_event[0])


def _enqueue_barrier(queue):
    _handle_error(_lib.enqueue_barrier(queue.ptr))

# }}}


# {{{ enqueue_migrate_mem_object*

def enqueue_migrate_mem_objects(queue, mem_objects, flags, wait_for=None):
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    c_mem_objs, num_mem_objs = _clobj_list(mem_objects)
    _handle_error(_lib.enqueue_migrate_mem_objects(
        _event, queue.ptr, c_mem_objs, num_mem_objs, flags,
        c_wait_for, num_wait_for))
    return Event._create(_event[0])


def enqueue_migrate_mem_object_ext(queue, mem_objects, flags, wait_for=None):
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    c_mem_objs, num_mem_objs = _clobj_list(mem_objects)
    _handle_error(_lib.enqueue_migrate_mem_object_ext(
        _event, queue.ptr, c_mem_objs, num_mem_objs, flags,
        c_wait_for, num_wait_for))
    return Event._create(_event[0])

# }}}


# {{{ _enqueue_wait_for_events

def _enqueue_wait_for_events(queue, wait_for=None):
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_wait_for_events(queue.ptr, c_wait_for,
                                               num_wait_for))

# }}}


# {{{ _enqueue_*_buffer

def _enqueue_read_buffer(queue, mem, hostbuf, device_offset=0,
                         wait_for=None, is_blocking=True):
    c_buf, size, _ = _c_buffer_from_obj(hostbuf, writable=True)
    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_read_buffer(
        ptr_event, queue.ptr, mem.ptr, c_buf, size, device_offset,
        c_wait_for, num_wait_for, bool(is_blocking),
        NannyEvent._handle(hostbuf)))
    return NannyEvent._create(ptr_event[0])


def _enqueue_write_buffer(queue, mem, hostbuf, device_offset=0,
                          wait_for=None, is_blocking=True):
    c_buf, size, c_ref = _c_buffer_from_obj(hostbuf, retain=True)
    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_write_buffer(
        ptr_event, queue.ptr, mem.ptr, c_buf, size, device_offset,
        c_wait_for, num_wait_for, bool(is_blocking),
        NannyEvent._handle(hostbuf, c_ref)))
    return NannyEvent._create(ptr_event[0])


def _enqueue_copy_buffer(queue, src, dst, byte_count=-1, src_offset=0,
                         dst_offset=0, wait_for=None):
    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_copy_buffer(
        ptr_event, queue.ptr, src.ptr, dst.ptr, byte_count, src_offset,
        dst_offset, c_wait_for, num_wait_for))
    return Event._create(ptr_event[0])


def _enqueue_read_buffer_rect(queue, mem, hostbuf, buffer_origin,
                              host_origin, region, buffer_pitches=None,
                              host_pitches=None, wait_for=None,
                              is_blocking=True):
    buffer_origin = tuple(buffer_origin)
    host_origin = tuple(host_origin)
    region = tuple(region)
    if buffer_pitches is None:
        buffer_pitches = _ffi.NULL
        buffer_pitches_l = 0
    else:
        buffer_pitches = tuple(buffer_pitches)
        buffer_pitches_l = len(buffer_pitches)
    if host_pitches is None:
        host_pitches = _ffi.NULL
        host_pitches_l = 0
    else:
        host_pitches = tuple(host_pitches)
        host_pitches_l = len(host_pitches)

    buffer_origin_l = len(buffer_origin)
    host_origin_l = len(host_origin)
    region_l = len(region)
    if (buffer_origin_l > 3 or host_origin_l > 3 or region_l > 3 or
            buffer_pitches_l > 2 or host_pitches_l > 2):
        raise RuntimeError("(buffer/host)_origin, (buffer/host)_pitches or "
                           "region has too many components",
                           status_code.INVALID_VALUE,
                           "enqueue_read_buffer_rect")
    c_buf, size, _ = _c_buffer_from_obj(hostbuf, writable=True)
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_read_buffer_rect(
        _event, queue.ptr, mem.ptr, c_buf, buffer_origin, buffer_origin_l,
        host_origin, host_origin_l, region, region_l, buffer_pitches,
        buffer_pitches_l, host_pitches, host_pitches_l, c_wait_for,
        num_wait_for, bool(is_blocking), NannyEvent._handle(hostbuf)))
    return NannyEvent._create(_event[0])


def _enqueue_write_buffer_rect(queue, mem, hostbuf, buffer_origin,
                               host_origin, region, buffer_pitches=None,
                               host_pitches=None, wait_for=None,
                               is_blocking=True):
    buffer_origin = tuple(buffer_origin)
    host_origin = tuple(host_origin)
    region = tuple(region)
    if buffer_pitches is None:
        buffer_pitches = _ffi.NULL
        buffer_pitches_l = 0
    else:
        buffer_pitches = tuple(buffer_pitches)
        buffer_pitches_l = len(buffer_pitches)
    if host_pitches is None:
        host_pitches = _ffi.NULL
        host_pitches_l = 0
    else:
        host_pitches = tuple(host_pitches)
        host_pitches_l = len(host_pitches)

    buffer_origin_l = len(buffer_origin)
    host_origin_l = len(host_origin)
    region_l = len(region)
    if (buffer_origin_l > 3 or host_origin_l > 3 or region_l > 3 or
            buffer_pitches_l > 2 or host_pitches_l > 2):
        raise RuntimeError("(buffer/host)_origin, (buffer/host)_pitches or "
                           "region has too many components",
                           status_code.INVALID_VALUE,
                           "enqueue_write_buffer_rect")
    c_buf, size, c_ref = _c_buffer_from_obj(hostbuf, retain=True)
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_write_buffer_rect(
        _event, queue.ptr, mem.ptr, c_buf, buffer_origin, buffer_origin_l,
        host_origin, host_origin_l, region, region_l, buffer_pitches,
        buffer_pitches_l, host_pitches, host_pitches_l, c_wait_for,
        num_wait_for, bool(is_blocking), NannyEvent._handle(hostbuf, c_ref)))
    return NannyEvent._create(_event[0])


def _enqueue_copy_buffer_rect(queue, src, dst, src_origin, dst_origin, region,
                              src_pitches=None, dst_pitches=None,
                              wait_for=None):
    src_origin = tuple(src_origin)
    dst_origin = tuple(dst_origin)
    region = tuple(region)
    if src_pitches is None:
        src_pitches = _ffi.NULL
        src_pitches_l = 0
    else:
        src_pitches = tuple(src_pitches)
        src_pitches_l = len(src_pitches)
    if dst_pitches is None:
        dst_pitches = _ffi.NULL
        dst_pitches_l = 0
    else:
        dst_pitches = tuple(dst_pitches)
        dst_pitches_l = len(dst_pitches)
    src_origin_l = len(src_origin)
    dst_origin_l = len(dst_origin)
    region_l = len(region)
    if (src_origin_l > 3 or dst_origin_l > 3 or region_l > 3 or
            src_pitches_l > 2 or dst_pitches_l > 2):
        raise RuntimeError("(src/dst)_origin, (src/dst)_pitches or "
                           "region has too many components",
                           status_code.INVALID_VALUE,
                           "enqueue_copy_buffer_rect")
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_copy_buffer_rect(
        _event, queue.ptr, src.ptr, dst.ptr, src_origin, src_origin_l,
        dst_origin, dst_origin_l, region, region_l, src_pitches,
        src_pitches_l, dst_pitches, dst_pitches_l, c_wait_for, num_wait_for))
    return Event._create(_event[0])


# PyPy bug report: https://bitbucket.org/pypy/pypy/issue/1777/unable-to-create-proper-numpy-array-from  # noqa
def enqueue_map_buffer(queue, buf, flags, offset, shape, dtype,
                       order="C", strides=None, wait_for=None,
                       is_blocking=True):
    dtype, shape, strides = _norm_shape_dtype(shape, dtype, order, strides,
                                              'enqueue_map_buffer')
    byte_size = dtype.itemsize
    for s in shape:
        byte_size *= s
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _event = _ffi.new('clobj_t*')
    _map = _ffi.new('clobj_t*')
    _handle_error(_lib.enqueue_map_buffer(_event, _map, queue.ptr, buf.ptr,
                                          flags, offset, byte_size, c_wait_for,
                                          num_wait_for, bool(is_blocking)))
    return (np.asarray(MemoryMap._create(_map[0], shape, dtype.str, strides)),
            Event._create(_event[0]))


def _enqueue_fill_buffer(queue, mem, pattern, offset, size, wait_for=None):
    c_pattern, psize, c_ref = _c_buffer_from_obj(pattern)
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_fill_buffer(
        _event, queue.ptr, mem.ptr, c_pattern, psize, offset, size,
        c_wait_for, num_wait_for))
    return Event._create(_event[0])

# }}}


# {{{ _enqueue_*_image

def _enqueue_read_image(queue, mem, origin, region, hostbuf, row_pitch=0,
                        slice_pitch=0, wait_for=None, is_blocking=True):
    origin = tuple(origin)
    region = tuple(region)
    origin_l = len(origin)
    region_l = len(region)
    if origin_l > 3 or region_l > 3:
        raise RuntimeError("origin or region has too many components",
                           status_code.INVALID_VALUE, "enqueue_read_image")
    c_buf, size, _ = _c_buffer_from_obj(hostbuf, writable=True)
    ptr_event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    # TODO check buffer size
    _handle_error(_lib.enqueue_read_image(
        ptr_event, queue.ptr, mem.ptr, origin, origin_l, region, region_l,
        c_buf, row_pitch, slice_pitch, c_wait_for, num_wait_for,
        bool(is_blocking), NannyEvent._handle(hostbuf)))
    return NannyEvent._create(ptr_event[0])


def _enqueue_copy_image(queue, src, dest, src_origin, dest_origin, region,
                        wait_for=None):
    src_origin = tuple(src_origin)
    region = tuple(region)
    src_origin_l = len(src_origin)
    dest_origin_l = len(dest_origin)
    region_l = len(region)
    if src_origin_l > 3 or dest_origin_l > 3 or region_l > 3:
        raise RuntimeError("(src/dest)origin or region has too many components",
                           status_code.INVALID_VALUE, "enqueue_copy_image")
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_copy_image(
        _event, queue.ptr, src.ptr, dest.ptr, src_origin, src_origin_l,
        dest_origin, dest_origin_l, region, region_l, c_wait_for, num_wait_for))
    return Event._create(_event[0])


def _enqueue_write_image(queue, mem, origin, region, hostbuf, row_pitch=0,
                         slice_pitch=0, wait_for=None, is_blocking=True):
    origin = tuple(origin)
    region = tuple(region)
    origin_l = len(origin)
    region_l = len(region)
    if origin_l > 3 or region_l > 3:
        raise RuntimeError("origin or region has too many components",
                           status_code.INVALID_VALUE, "enqueue_write_image")
    c_buf, size, c_ref = _c_buffer_from_obj(hostbuf, retain=True)
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    # TODO: check buffer size
    _handle_error(_lib.enqueue_write_image(
        _event, queue.ptr, mem.ptr, origin, origin_l, region, region_l,
        c_buf, row_pitch, slice_pitch, c_wait_for, num_wait_for,
        bool(is_blocking), NannyEvent._handle(hostbuf, c_ref)))
    return NannyEvent._create(_event[0])


def enqueue_map_image(queue, img, flags, origin, region, shape, dtype,
                      order="C", strides=None, wait_for=None, is_blocking=True):
    origin = tuple(origin)
    region = tuple(region)
    origin_l = len(origin)
    region_l = len(region)
    if origin_l > 3 or region_l > 3:
        raise RuntimeError("origin or region has too many components",
                           status_code.INVALID_VALUE, "enqueue_map_image")
    dtype, shape, strides = _norm_shape_dtype(shape, dtype, order, strides,
                                              'enqueue_map_image')
    _event = _ffi.new('clobj_t*')
    _map = _ffi.new('clobj_t*')
    _row_pitch = _ffi.new('size_t*')
    _slice_pitch = _ffi.new('size_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_map_image(_event, _map, queue.ptr, img.ptr,
                                         flags, origin, origin_l, region,
                                         region_l, _row_pitch, _slice_pitch,
                                         c_wait_for, num_wait_for, is_blocking))
    return (np.asarray(MemoryMap._create(_map[0], shape, dtype.str, strides)),
            Event._create(_event[0]), _row_pitch[0], _slice_pitch[0])


def enqueue_fill_image(queue, img, color, origin, region, wait_for=None):
    origin = tuple(origin)
    region = tuple(region)
    origin_l = len(origin)
    region_l = len(region)
    color_l = len(color)
    if origin_l > 3 or region_l > 3 or color_l > 4:
        raise RuntimeError("origin, region or color has too many components",
                           status_code.INVALID_VALUE, "enqueue_fill_image")
    color = np.array(color).astype(img._fill_type)
    c_color = _ffi.cast('void*', color.__array_interface__['data'][0])
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_fill_image(_event, queue.ptr, img.ptr,
                                          c_color, origin, origin_l, region,
                                          region_l, c_wait_for, num_wait_for))
    return Event._create(_event[0])


def _enqueue_copy_image_to_buffer(queue, src, dest, origin, region, offset,
                                  wait_for=None):
    origin = tuple(origin)
    region = tuple(region)
    origin_l = len(origin)
    region_l = len(region)
    if origin_l > 3 or region_l > 3:
        raise RuntimeError("origin or region has too many components",
                           status_code.INVALID_VALUE,
                           "enqueue_copy_image_to_buffer")
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_copy_image_to_buffer(
        _event, queue.ptr, src.ptr, dest.ptr, origin, origin_l, region,
        region_l, offset, c_wait_for, num_wait_for))
    return Event._create(_event[0])


def _enqueue_copy_buffer_to_image(queue, src, dest, offset, origin, region,
                                  wait_for=None):
    origin = tuple(origin)
    region = tuple(region)
    origin_l = len(origin)
    region_l = len(region)
    if origin_l > 3 or region_l > 3:
        raise RuntimeError("origin or region has too many components",
                           status_code.INVALID_VALUE,
                           "enqueue_copy_buffer_to_image")
    _event = _ffi.new('clobj_t*')
    c_wait_for, num_wait_for = _clobj_list(wait_for)
    _handle_error(_lib.enqueue_copy_buffer_to_image(
        _event, queue.ptr, src.ptr, dest.ptr, offset, origin, origin_l,
        region, region_l, c_wait_for, num_wait_for))
    return Event._create(_event[0])

# }}}


# {{{ gl interop

def have_gl():
    return bool(_lib.have_gl())


class _GLObject(object):
    def get_gl_object_info(self):
        otype = _ffi.new('cl_gl_object_type*')
        gl_name = _ffi.new('GLuint*')
        _handle_error(_lib.get_gl_object_info(self.ptr, otype, gl_name))
        return otype[0], gl_name[0]


class GLBuffer(MemoryObject, _GLObject):
    _id = 'gl_buffer'

    def __init__(self, context, flags, bufobj):
        MemoryObject.__init__(self)
        ptr = _ffi.new('clobj_t*')
        _handle_error(_lib.create_from_gl_buffer(
            ptr, context.ptr, flags, bufobj))
        self.ptr = ptr[0]


class GLRenderBuffer(MemoryObject, _GLObject):
    _id = 'gl_renderbuffer'

    def __init__(self, context, flags, bufobj):
        MemoryObject.__init__(self, bufobj)
        c_buf, bufsize, retained = self._handle_buf_flags(flags)
        ptr = _ffi.new('clobj_t*')
        _handle_error(_lib.create_from_gl_renderbuffer(
            ptr, context.ptr, flags, c_buf))
        self.ptr = ptr[0]


def _create_gl_enqueue(what):
    def enqueue_gl_objects(queue, mem_objects, wait_for=None):
        ptr_event = _ffi.new('clobj_t*')
        c_wait_for, num_wait_for = _clobj_list(wait_for)
        c_mem_objects, num_mem_objects = _clobj_list(mem_objects)
        _handle_error(what(ptr_event, queue.ptr, c_mem_objects, num_mem_objects,
                           c_wait_for, num_wait_for))
        return Event._create(ptr_event[0])
    return enqueue_gl_objects

if _lib.have_gl():
    enqueue_acquire_gl_objects = _create_gl_enqueue(
        _lib.enqueue_acquire_gl_objects)
    enqueue_release_gl_objects = _create_gl_enqueue(
        _lib.enqueue_release_gl_objects)
    try:
        get_apple_cgl_share_group = _lib.get_apple_cgl_share_group
    except AttributeError:
        pass

# }}}


def _cffi_property(_name=None, read=True, write=True):
    def _deco(get_ptr):
        name = _name if _name else get_ptr.__name__
        return property((lambda self: getattr(get_ptr(self), name)) if read
                        else (lambda self: None),
                        (lambda self, v: setattr(get_ptr(self), name, v))
                        if write else (lambda self, v: None))
    return _deco


# {{{ ImageFormat

class ImageFormat(object):
    # Hack around fmt.__dict__ check in test_wrapper.py
    __dict__ = {}
    __slots__ = ('ptr',)

    def __init__(self, channel_order=0, channel_type=0):
        self.ptr = _ffi.new("cl_image_format*")
        self.channel_order = channel_order
        self.channel_data_type = channel_type

    @_cffi_property('image_channel_order')
    def channel_order(self):
        return self.ptr

    @_cffi_property('image_channel_data_type')
    def channel_data_type(self):
        return self.ptr

    @property
    def channel_count(self):
        try:
            return {
                channel_order.R: 1,
                channel_order.A: 1,
                channel_order.RG: 2,
                channel_order.RA: 2,
                channel_order.RGB: 3,
                channel_order.RGBA: 4,
                channel_order.BGRA: 4,
                channel_order.INTENSITY: 1,
                channel_order.LUMINANCE: 1,
            }[self.channel_order]
        except KeyError:
            raise LogicError("unrecognized channel order",
                             status_code.INVALID_VALUE,
                             "ImageFormat.channel_count")

    @property
    def dtype_size(self):
        try:
            return {
                channel_type.SNORM_INT8: 1,
                channel_type.SNORM_INT16: 2,
                channel_type.UNORM_INT8: 1,
                channel_type.UNORM_INT16: 2,
                channel_type.UNORM_SHORT_565: 2,
                channel_type.UNORM_SHORT_555: 2,
                channel_type.UNORM_INT_101010: 4,
                channel_type.SIGNED_INT8: 1,
                channel_type.SIGNED_INT16: 2,
                channel_type.SIGNED_INT32: 4,
                channel_type.UNSIGNED_INT8: 1,
                channel_type.UNSIGNED_INT16: 2,
                channel_type.UNSIGNED_INT32: 4,
                channel_type.HALF_FLOAT: 2,
                channel_type.FLOAT: 4,
            }[self.channel_data_type]
        except KeyError:
            raise LogicError("unrecognized channel data type",
                             status_code.INVALID_VALUE,
                             "ImageFormat.channel_dtype_size")

    @property
    def itemsize(self):
        return self.channel_count * self.dtype_size

    def __repr__(self):
        return "ImageFormat(%s, %s)" % (
                channel_order.to_string(self.channel_order,
                    "<unknown channel order 0x%x>"),
                channel_type.to_string(self.channel_data_type,
                    "<unknown channel data type 0x%x>"))

    def __eq__(self, other):
        return (self.channel_order == other.channel_order
                and self.channel_data_type == other.channel_data_type)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((type(self), self.channel_order, self.channel_data_type))


def get_supported_image_formats(context, flags, image_type):
    info = _ffi.new('generic_info*')
    _handle_error(_lib.context__get_supported_image_formats(
        context.ptr, flags, image_type, info))
    return _generic_info_to_python(info)

# }}}


# {{{ ImageDescriptor

def _write_only_property(*arg):
    return property().setter(*arg)


class ImageDescriptor(object):
    __slots__ = ('ptr',)

    def __init__(self):
        self.ptr = _ffi.new("cl_image_desc*")

    @_cffi_property()
    def image_type(self):
        return self.ptr

    @_cffi_property('image_array_size')
    def array_size(self):
        return self.ptr

    @_cffi_property()
    def num_mip_levels(self):
        return self.ptr

    @_cffi_property()
    def num_samples(self):
        return self.ptr

    @_write_only_property
    def shape(self, shape):
        l = len(shape)
        if l > 3:
            raise LogicError("shape has too many components",
                             status_code.INVALID_VALUE, "transfer")
        desc = self.ptr
        desc.image_width = shape[0] if l > 0 else 1
        desc.image_height = shape[1] if l > 1 else 1
        desc.image_depth = shape[2] if l > 2 else 1
        desc.image_array_size = desc.image_depth

    @_write_only_property
    def pitches(self, pitches):
        l = len(pitches)
        if l > 2:
            raise LogicError("pitches has too many components",
                             status_code.INVALID_VALUE, "transfer")
        desc = self.ptr
        desc.image_row_pitch = pitches[0] if l > 0 else 1
        desc.image_slice_pitch = pitches[1] if l > 1 else 1

    @_write_only_property
    def buffer(self, buff):
        self.ptr.buffer = buff.ptr.int_ptr if buff else _ffi.NULL

# }}}


# {{{ Image

_int_dtype = ({
    8: np.int64,
    4: np.int32,
    2: np.int16,
    1: np.int8,
})[_ffi.sizeof('int')]

_uint_dtype = ({
    8: np.uint64,
    4: np.uint32,
    2: np.uint16,
    1: np.uint8,
})[_ffi.sizeof('unsigned')]

_float_dtype = ({
    8: np.float64,
    4: np.float32,
    2: np.float16,
})[_ffi.sizeof('float')]

_fill_dtype_dict = {
    _lib.TYPE_INT: _int_dtype,
    _lib.TYPE_UINT: _uint_dtype,
    _lib.TYPE_FLOAT: _float_dtype,
    }


class Image(MemoryObject):
    _id = 'image'

    def __init_dispatch(self, *args):
        if len(args) == 5:
            # >= 1.2
            self.__init_1_2(*args)
        elif len(args) == 6:
            # <= 1.1
            self.__init_legacy(*args)
        else:
            assert False
        self._fill_type = _fill_dtype_dict[_lib.image__get_fill_type(self.ptr)]

    def __init_1_2(self, context, flags, fmt, desc, hostbuf):
        MemoryObject.__init__(self, hostbuf)
        c_buf, size, retained_buf = self._handle_buf_flags(flags)
        ptr = _ffi.new('clobj_t*')
        _handle_error(_lib.create_image_from_desc(ptr, context.ptr, flags,
                                                  fmt.ptr, desc.ptr, c_buf))
        self.ptr = ptr[0]

    def __init_legacy(self, context, flags, fmt, shape, pitches, hostbuf):
        if shape is None:
            raise LogicError("'shape' must be given",
                             status_code.INVALID_VALUE, "Image")
        MemoryObject.__init__(self, hostbuf)
        c_buf, size, retained_buf = self._handle_buf_flags(flags)
        dims = len(shape)
        if dims == 2:
            width, height = shape
            pitch = 0
            if pitches is not None:
                try:
                    pitch, = pitches
                except ValueError:
                    raise LogicError("invalid length of pitch tuple",
                                     status_code.INVALID_VALUE, "Image")

            # check buffer size
            if (hostbuf is not None and
                    max(pitch, width * fmt.itemsize) * height > size):
                raise LogicError("buffer too small",
                                 status_code.INVALID_VALUE, "Image")

            ptr = _ffi.new('clobj_t*')
            _handle_error(_lib.create_image_2d(ptr, context.ptr, flags, fmt.ptr,
                                               width, height, pitch, c_buf))
            self.ptr = ptr[0]
        elif dims == 3:
            width, height, depth = shape
            pitch_x, pitch_y = 0, 0
            if pitches is not None:
                try:
                    pitch_x, pitch_y = pitches
                except ValueError:
                    raise LogicError("invalid length of pitch tuple",
                                     status_code.INVALID_VALUE, "Image")

            # check buffer size
            if (hostbuf is not None and
                (max(max(pitch_x, width * fmt.itemsize) *
                     height, pitch_y) * depth > size)):
                raise LogicError("buffer too small",
                                 status_code.INVALID_VALUE, "Image")

            ptr = _ffi.new('clobj_t*')
            _handle_error(_lib.create_image_3d(
                ptr, context.ptr, flags, fmt.ptr,
                width, height, depth, pitch_x, pitch_y, c_buf))

            self.ptr = ptr[0]
        else:
            raise LogicError("invalid dimension",
                             status_code.INVALID_VALUE, "Image")

    def __init__(self, context, flags, format, shape=None, pitches=None,
            hostbuf=None, is_array=False, buffer=None):

        if shape is None and hostbuf is None:
            raise Error("'shape' must be passed if 'hostbuf' is not given")

        if shape is None and hostbuf is not None:
            shape = hostbuf.shape

        if hostbuf is not None and not \
                (flags & (mem_flags.USE_HOST_PTR | mem_flags.COPY_HOST_PTR)):
            from warnings import warn
            warn("'hostbuf' was passed, but no memory flags to make use of it.")

        if hostbuf is None and pitches is not None:
            raise Error("'pitches' may only be given if 'hostbuf' is given")

        if context._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
            if buffer is not None and is_array:
                    raise ValueError(
                            "'buffer' and 'is_array' are mutually exclusive")

            if len(shape) == 3:
                if buffer is not None:
                    raise TypeError(
                            "'buffer' argument is not supported for 3D arrays")
                elif is_array:
                    image_type = mem_object_type.IMAGE2D_ARRAY
                else:
                    image_type = mem_object_type.IMAGE3D

            elif len(shape) == 2:
                if buffer is not None:
                    raise TypeError(
                            "'buffer' argument is not supported for 2D arrays")
                elif is_array:
                    image_type = mem_object_type.IMAGE1D_ARRAY
                else:
                    image_type = mem_object_type.IMAGE2D

            elif len(shape) == 1:
                if buffer is not None:
                    image_type = mem_object_type.IMAGE1D_BUFFER
                elif is_array:
                    raise TypeError("array of zero-dimensional images not supported")
                else:
                    image_type = mem_object_type.IMAGE1D

            else:
                raise ValueError("images cannot have more than three dimensions")

            desc = ImageDescriptor()

            desc.image_type = image_type
            desc.shape = shape  # also sets desc.array_size

            if pitches is None:
                desc.pitches = (0, 0)
            else:
                desc.pitches = pitches

            desc.num_mip_levels = 0  # per CL 1.2 spec
            desc.num_samples = 0  # per CL 1.2 spec
            desc.buffer = buffer

            self.__init_dispatch(context, flags, format, desc, hostbuf)
        else:
            # legacy init for CL 1.1 and older
            if is_array:
                raise TypeError("'is_array=True' is not supported for CL < 1.2")
            # if num_mip_levels is not None:
                # raise TypeError(
                #       "'num_mip_levels' argument is not supported for CL < 1.2")
            # if num_samples is not None:
                # raise TypeError(
                #        "'num_samples' argument is not supported for CL < 1.2")
            if buffer is not None:
                raise TypeError("'buffer' argument is not supported for CL < 1.2")

            self.__init_dispatch(context, flags, format, shape,
                    pitches, hostbuf)

    def get_image_info(self, param):
        info = _ffi.new('generic_info*')
        _handle_error(_lib.image__get_image_info(self.ptr, param, info))
        return _generic_info_to_python(info)

    @property
    def shape(self):
        if self.type == mem_object_type.IMAGE2D:
            return (self.width, self.height)
        elif self.type == mem_object_type.IMAGE3D:
            return (self.width, self.height, self.depth)
        else:
            raise LogicError("only images have shapes")


class _ImageInfoGetter:
    def __init__(self, event):
        from warnings import warn
        warn("Image.image.attr is deprecated. "
                "Use Image.attr directly, instead.")

        self.event = event

    def __getattr__(self, name):
        try:
            inf_attr = getattr(image_info, name.upper())
        except AttributeError:
            raise AttributeError("%s has no attribute '%s'"
                    % (type(self), name))
        else:
            return self.event.get_image_info(inf_attr)

Image.image = property(_ImageInfoGetter)

# }}}


# {{{ Sampler

class Sampler(_Common, _CLKernelArg):
    _id = 'sampler'

    def __init__(self, context, normalized_coords, addressing_mode, filter_mode):
        ptr = _ffi.new('clobj_t*')
        _handle_error(_lib.create_sampler(
            ptr, context.ptr, normalized_coords, addressing_mode, filter_mode))
        self.ptr = ptr[0]

# }}}


# {{{ GLTexture

class GLTexture(Image, _GLObject):
    _id = 'gl_texture'

    def __init__(self, context, flags, texture_target, miplevel, texture, dims=None):
        ptr = _ffi.new('clobj_t*')
        _handle_error(_lib.create_from_gl_texture(
            ptr, context.ptr, flags, texture_target, miplevel, texture))
        self.ptr = ptr[0]

# }}}


# {{{ DeviceTopologyAmd

class DeviceTopologyAmd(object):
    # Hack around fmt.__dict__ check in test_wrapper.py
    __dict__ = {}
    __slots__ = ('ptr',)

    def __init__(self, bus=0, device=0, function=0):
        self.ptr = _ffi.new("cl_device_topology_amd*")
        self.bus = bus
        self.device = device
        self.function = function

    def _check_range(self, value, prop=None):
        if (value < -127) or (value > 127):
            raise ValueError("Value %s not in range [-127, 127].")

    @_cffi_property('pcie')
    def _pcie(self):
        return self.ptr

    @property
    def bus(self):
        return self._pcie.bus

    @bus.setter
    def bus(self, value):
        self._check_range(value)
        self._pcie.bus = value

    @property
    def device(self):
        return self._pcie.device

    @device.setter
    def device(self, value):
        self._pcie.device = value

    @property
    def function(self):
        return self._pcie.function

    @function.setter
    def function(self, value):
        self._pcie.function = value

# }}}


# {{{ get_info monkeypatchery

def add_get_info_attrs(cls, info_method, info_class, cacheable_attrs=None):
    if cacheable_attrs is None:
        cacheable_attrs = []

    def make_getinfo(info_method, info_name, info_attr):
        def result(self):
            return info_method(self, info_attr)

        return property(result)

    def make_cacheable_getinfo(info_method, info_name, cache_attr, info_attr):
        def result(self):
            try:
                return getattr(self, cache_attr)
            except AttributeError:
                pass

            result = info_method(self, info_attr)
            setattr(self, cache_attr, result)
            return result

        return property(result)

    for info_name, info_value in six.iteritems(info_class.__dict__):
        if info_name == "to_string" or info_name.startswith("_"):
            continue

        info_lower = info_name.lower()
        info_constant = getattr(info_class, info_name)
        if info_name in cacheable_attrs:
            cache_attr = intern("_info_cache_"+info_lower)
            setattr(cls, info_lower, make_cacheable_getinfo(
                info_method, info_lower, cache_attr, info_constant))
        else:
            setattr(cls, info_lower, make_getinfo(
                    info_method, info_name, info_constant))

add_get_info_attrs(Platform, Platform.get_info, platform_info),
add_get_info_attrs(Device, Device.get_info, device_info,
                ["PLATFORM", "MAX_WORK_GROUP_SIZE", "MAX_COMPUTE_UNITS"])
add_get_info_attrs(Context, Context.get_info, context_info)
add_get_info_attrs(CommandQueue, CommandQueue.get_info, command_queue_info,
                ["CONTEXT", "DEVICE"])
add_get_info_attrs(Event, Event.get_info, event_info)
add_get_info_attrs(MemoryObjectHolder, MemoryObjectHolder.get_info, mem_info)
add_get_info_attrs(Image, Image.get_image_info, image_info)
add_get_info_attrs(Kernel, Kernel.get_info, kernel_info)
add_get_info_attrs(Sampler, Sampler.get_info, sampler_info)

# }}}


if have_gl():
    def gl_object_get_gl_object(self):
        return self.get_gl_object_info()[1]

    GLBuffer.gl_object = property(gl_object_get_gl_object)
    GLTexture.gl_object = property(gl_object_get_gl_object)

# vim: foldmethod=marker
