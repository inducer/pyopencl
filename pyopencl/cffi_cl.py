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
from six.moves import map, range, zip

import warnings
import numpy as np
import sys

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

try:
    _unicode = eval('unicode')
    _ffi_pystr = _ffi.string
except:
    _unicode = str
    _bytes = bytes

    def _ffi_pystr(s):
        return _ffi.string(s).decode()
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
        return other == self.int_ptr

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


class _NoInit(object):
    def __init__(self):
        raise RuntimeError("This class cannot be instantiated.")


# {{{ constant classes

# /!\ If you add anything here, add it to pyopencl/__init__.py as well.

class program_kind(_NoInit):  # noqa
    pass


class status_code(_NoInit):  # noqa
    pass


class platform_info(_NoInit):  # noqa
    pass


class device_type(_NoInit):  # noqa
    pass


class device_info(_NoInit):  # noqa
    pass


class device_fp_config(_NoInit):  # noqa
    pass


class device_mem_cache_type(_NoInit):  # noqa
    pass


class device_local_mem_type(_NoInit):  # noqa
    pass


class device_exec_capabilities(_NoInit):  # noqa
    pass


class device_svm_capabilities(_NoInit):  # noqa
    pass


class command_queue_properties(_NoInit):  # noqa
    pass


class context_info(_NoInit):  # noqa
    pass


class gl_context_info(_NoInit):  # noqa
    pass


class context_properties(_NoInit):  # noqa
    pass


class command_queue_info(_NoInit):  # noqa
    pass


class queue_properties(_NoInit):  # noqa
    pass


class mem_flags(_NoInit):  # noqa
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


class svm_mem_flags(_NoInit):  # noqa
    pass


class channel_order(_NoInit):  # noqa
    pass


class channel_type(_NoInit):  # noqa
    pass


class mem_object_type(_NoInit):  # noqa
    pass


class mem_info(_NoInit):  # noqa
    pass


class image_info(_NoInit):  # noqa
    pass


class addressing_mode(_NoInit):  # noqa
    pass


class filter_mode(_NoInit):  # noqa
    pass


class sampler_info(_NoInit):  # noqa
    pass


class map_flags(_NoInit):  # noqa
    pass


class program_info(_NoInit):  # noqa
    pass


class program_build_info(_NoInit):  # noqa
    pass


class program_binary_type(_NoInit):  # noqa
    pass


class kernel_info(_NoInit):  # noqa
    pass


class kernel_arg_info(_NoInit):  # noqa
    pass


class kernel_arg_address_qualifier(_NoInit):  # noqa
    pass


class kernel_arg_access_qualifier(_NoInit):  # noqa
    pass


class kernel_arg_type_qualifier(_NoInit):  # noqa
    pass


class kernel_work_group_info(_NoInit):  # noqa
    pass


class event_info(_NoInit):  # noqa
    pass


class command_type(_NoInit):  # noqa
    pass


class command_execution_status(_NoInit):  # noqa
    pass


class profiling_info(_NoInit):  # noqa
    pass


class mem_migration_flags(_NoInit):  # noqa
    pass


class mem_migration_flags_ext(_NoInit):  # noqa
    pass


class device_partition_property(_NoInit):  # noqa
    pass


class device_affinity_domain(_NoInit):  # noqa
    pass


class gl_object_type(_NoInit):  # noqa
    pass


class gl_texture_info(_NoInit):  # noqa
    pass


class migrate_mem_object_flags_ext(_NoInit):  # noqa
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


class MemoryError(Error):
    pass


class LogicError(Error):
    pass


class RuntimeError(Error):
    pass


def _handle_error(error):
    if error == _ffi.NULL:
        return
    if error.other == 1:
        # non-pyopencl exceptions are handled here
        import exceptions
        e = exceptions.RuntimeError(_ffi_pystr(error.msg))
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

# }}}


# {{{ memory objects

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


class LocalMemory(_CLKernelArg):
    __slots__ = ('_size',)

    def __init__(self, size):
        self._size = size

    @property
    def size(self):
        return self._size


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


class MemoryMap(_Common):
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

    def release(self, queue=None, wait_for=None):
        c_wait_for, num_wait_for = _clobj_list(wait_for)
        _event = _ffi.new('clobj_t*')
        _handle_error(_lib.memory_map__release(
            self.ptr, queue.ptr if queue is not None else _ffi.NULL,
            c_wait_for, num_wait_for, _event))
        return Event._create(_event[0])


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
        # {{{ fall back to the old CPython buffer protocol API

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

# }}}


# {{{ Program

class _Program(_Common):
    _id = 'program'

    def __init__(self, *args):
        if len(args) == 2:
            self._init_source(*args)
        else:
            self._init_binary(*args)

    def _init_source(self, context, src):
        ptr_program = _ffi.new('clobj_t*')
        _handle_error(_lib.create_program_with_source(
            ptr_program, context.ptr, _to_cstring(src)))
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

# }}}


# {{{ Kernel

class Kernel(_Common):
    _id = 'kernel'

    def __init__(self, program, name):
        ptr_kernel = _ffi.new('clobj_t*')
        _handle_error(_lib.create_kernel(ptr_kernel, program.ptr,
                                         _to_cstring(name)))
        self.ptr = ptr_kernel[0]

    def set_arg(self, arg_index, arg):
        # If you change this, also change the kernel call generation logic.
        if arg is None:
            _handle_error(_lib.kernel__set_arg_null(self.ptr, arg_index))
        elif isinstance(arg, _CLKernelArg):
            if isinstance(arg, MemoryObjectHolder):
                _handle_error(_lib.kernel__set_arg_mem(self.ptr, arg_index, arg.ptr))
            elif isinstance(arg, Sampler):
                _handle_error(_lib.kernel__set_arg_sampler(self.ptr, arg_index,
                                                           arg.ptr))
            elif isinstance(arg, LocalMemory):
                _handle_error(_lib.kernel__set_arg_buf(self.ptr, arg_index,
                                                       _ffi.NULL, arg.size))
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
        info = _ffi.new('generic_info*')
        _handle_error(_lib.kernel__get_work_group_info(
            self.ptr, param, device.ptr, info))
        return _generic_info_to_python(info)

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
    if host_pitches is None:
        host_pitches = _ffi.NULL
        host_pitches_l = 0
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
    if host_pitches is None:
        host_pitches = _ffi.NULL
        host_pitches_l = 0
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
        src_pitches_l = len(src_pitches)
    if dst_pitches is None:
        dst_pitches = _ffi.NULL
        dst_pitches_l = 0
    else:
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

    def __init__(self, *args):
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

    def get_image_info(self, param):
        info = _ffi.new('generic_info*')
        _handle_error(_lib.image__get_image_info(self.ptr, param, info))
        return _generic_info_to_python(info)

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

# vim: foldmethod=marker
