# -*- coding: utf-8 -*-

from __future__ import division, absolute_import, print_function

__copyright__ = "Copyright (C) 2009-15 Andreas Kloeckner"

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

import re
import six
from six.moves import input

from pyopencl.version import VERSION, VERSION_STATUS, VERSION_TEXT  # noqa

try:
    import pyopencl.cffi_cl as _cl
except ImportError:
    import os
    from os.path import dirname, join, realpath
    if realpath(join(os.getcwd(), "pyopencl")) == realpath(dirname(__file__)):
        from warnings import warn
        warn("It looks like you are importing PyOpenCL from "
                "its source directory. This likely won't work.")
    raise

import numpy as np

from pyopencl.cffi_cl import (  # noqa
        get_cl_header_version,
        program_kind,
        status_code,
        platform_info,
        device_type,
        device_info,
        device_fp_config,
        device_mem_cache_type,
        device_local_mem_type,
        device_exec_capabilities,
        device_svm_capabilities,

        command_queue_properties,
        context_info,
        gl_context_info,
        context_properties,
        command_queue_info,
        queue_properties,

        mem_flags,
        svm_mem_flags,

        channel_order,
        channel_type,
        mem_object_type,
        mem_info,
        image_info,
        addressing_mode,
        filter_mode,
        sampler_info,
        map_flags,
        program_info,
        program_build_info,
        program_binary_type,

        kernel_info,
        kernel_arg_info,
        kernel_arg_address_qualifier,
        kernel_arg_access_qualifier,
        kernel_arg_type_qualifier,
        kernel_work_group_info,

        event_info,
        command_type,
        command_execution_status,
        profiling_info,
        mem_migration_flags,
        mem_migration_flags_ext,
        device_partition_property,
        device_affinity_domain,
        gl_object_type,
        gl_texture_info,
        migrate_mem_object_flags_ext,

        Error, MemoryError, LogicError, RuntimeError,

        Platform,
        get_platforms,
        unload_platform_compiler,

        Device,
        Context,
        CommandQueue,
        LocalMemory,
        MemoryObjectHolder,
        MemoryObject,
        MemoryMap,
        Buffer,
        SVMAllocation,
        SVM,
        SVMMap,

        CompilerWarning,
        _Program,
        Kernel,

        Event,
        wait_for_events,
        NannyEvent,
        UserEvent,

        enqueue_nd_range_kernel,
        enqueue_task,

        _enqueue_marker_with_wait_list,
        _enqueue_marker,
        _enqueue_barrier_with_wait_list,

        enqueue_migrate_mem_objects,
        enqueue_migrate_mem_object_ext,

        _enqueue_barrier_with_wait_list,
        _enqueue_read_buffer,
        _enqueue_write_buffer,
        _enqueue_copy_buffer,
        _enqueue_read_buffer_rect,
        _enqueue_write_buffer_rect,
        _enqueue_copy_buffer_rect,

        enqueue_map_buffer,
        _enqueue_fill_buffer,
        _enqueue_read_image,
        _enqueue_copy_image,
        _enqueue_write_image,
        enqueue_map_image,
        enqueue_fill_image,
        _enqueue_copy_image_to_buffer,
        _enqueue_copy_buffer_to_image,
        enqueue_svm_memfill,
        enqueue_svm_migratemem,

        have_gl,
        _GLObject,
        GLBuffer,
        GLRenderBuffer,

        ImageFormat,
        get_supported_image_formats,

        ImageDescriptor,
        Image,
        Sampler,
        GLTexture,
        DeviceTopologyAmd,

        add_get_info_attrs as _add_get_info_attrs,
        )

if _cl.have_gl():
    try:
        from pyopencl.cffi_cl import get_apple_cgl_share_group  # noqa
    except ImportError:
        pass

    try:
        from pyopencl.cffi_cl import (  # noqa
            enqueue_acquire_gl_objects,
            enqueue_release_gl_objects,
        )
    except ImportError:
        pass


# {{{ find pyopencl shipped source code

def _find_pyopencl_include_path():
    from pkg_resources import Requirement, resource_filename, DistributionNotFound
    try:
        # Try to find the resource with pkg_resources (the recommended
        # setuptools approach)
        return resource_filename(Requirement.parse("pyopencl2"), "pyopencl/cl")
    except DistributionNotFound:
        # If pkg_resources can't find it (e.g. if the module is part of a
        # frozen application), try to find the include path in the same
        # directory as this file
        from os.path import join, abspath, dirname, exists

        include_path = join(abspath(dirname(__file__)), "cl")
        # If that doesn't exist, just re-raise the exception caught from
        # resource_filename.
        if not exists(include_path):
            raise

        return include_path

# }}}


# {{{ Program (wrapper around _Program, adds caching support)

_DEFAULT_BUILD_OPTIONS = []
_DEFAULT_INCLUDE_OPTIONS = ["-I", _find_pyopencl_include_path()]

# map of platform.name to build options list
_PLAT_BUILD_OPTIONS = {}


def enable_debugging(platform_or_context):
    """Enables debugging for all code subsequently compiled by
    PyOpenCL on the passed *platform*. Alternatively, a context
    may be passed.
    """

    if isinstance(platform_or_context, Context):
        platform = platform_or_context.devices[0].platform
    else:
        platform = platform_or_context

    if "AMD Accelerated" in platform.name:
        _PLAT_BUILD_OPTIONS.setdefault(platform.name, []).extend(
                ["-g", "-O0"])
        import os
        os.environ["CPU_MAX_COMPUTE_UNITS"] = "1"
    else:
        from warnings import warn
        warn("do not know how to enable debugging on '%s'"
                % platform.name)


class Program(object):
    def __init__(self, arg1, arg2=None, arg3=None):
        if arg2 is None:
            # 1-argument form: program
            self._prg = arg1

        elif arg3 is None:
            # 2-argument form: context, source
            context, source = arg1, arg2

            from pyopencl.tools import is_spirv
            if is_spirv(source):
                # no caching in SPIR-V case
                self._context = context
                self._prg = _cl._Program(context, source)
                return

            import sys
            if isinstance(source, six.text_type) and sys.version_info < (3,):
                from warnings import warn
                warn("Received OpenCL source code in Unicode, "
                     "should be ASCII string. Attempting conversion.",
                     stacklevel=2)
                source = source.encode()

            self._context = context
            self._source = source
            self._prg = None

        else:
            context, device, binaries = arg1, arg2, arg3
            self._context = context
            self._prg = _cl._Program(context, device, binaries)

    def _get_prg(self):
        if self._prg is not None:
            return self._prg
        else:
            # "no program" can only happen in from-source case.
            from warnings import warn
            warn("Pre-build attribute access defeats compiler caching.",
                    stacklevel=3)

            self._prg = _cl._Program(self._context, self._source)
            del self._context
            return self._prg

    def get_info(self, arg):
        return self._get_prg().get_info(arg)

    def get_build_info(self, *args, **kwargs):
        return self._get_prg().get_build_info(*args, **kwargs)

    def all_kernels(self):
        return self._get_prg().all_kernels()

    def int_ptr(self):
        return self._get_prg().int_ptr
    int_ptr = property(int_ptr, doc=_cl._Program.int_ptr.__doc__)

    def from_int_ptr(int_ptr_value):
        return Program(_cl._Program.from_int_ptr(int_ptr_value))
    from_int_ptr.__doc__ = _cl._Program.from_int_ptr.__doc__
    from_int_ptr = staticmethod(from_int_ptr)

    def __getattr__(self, attr):
        try:
            knl = Kernel(self, attr)
            # Nvidia does not raise errors even for invalid names,
            # but this will give an error if the kernel is invalid.
            knl.num_args
            knl._source = getattr(self, "_source", None)
            return knl
        except LogicError:
            raise AttributeError("'%s' was not found as a program "
                    "info attribute or as a kernel name" % attr)

    # {{{ build

    if six.PY3:
        _find_unsafe_re_opts = re.ASCII
    else:
        _find_unsafe_re_opts = 0

    _find_unsafe = re.compile(br'[^\w@%+=:,./-]', _find_unsafe_re_opts).search

    @classmethod
    def _shlex_quote(cls, s):
        """Return a shell-escaped version of the string *s*."""

        # Stolen from https://hg.python.org/cpython/file/default/Lib/shlex.py#l276

        if not s:
            return "''"

        if cls._find_unsafe(s) is None:
            return s

        # use single quotes, and put single quotes into double quotes
        # the string $'b is then quoted as '$'"'"'b'
        import sys
        if sys.platform.startswith("win"):
            # not sure how to escape that
            assert b'"' not in s
            return b'"' + s + b'"'
        else:
            return b"'" + s.replace(b"'", b"'\"'\"'") + b"'"

    @classmethod
    def _process_build_options(cls, context, options):
        if isinstance(options, six.string_types):
            import shlex
            if six.PY2:
                # shlex.split takes bytes (py2 str) on py2
                if isinstance(options, six.text_type):
                    options = options.encode("utf-8")
            else:
                # shlex.split takes unicode (py3 str) on py3
                if isinstance(options, six.binary_type):
                    options = options.decode("utf-8")

            options = shlex.split(options)

        def encode_if_necessary(s):
            if isinstance(s, six.text_type):
                return s.encode("utf-8")
            else:
                return s

        options = (options
                + _DEFAULT_BUILD_OPTIONS
                + _DEFAULT_INCLUDE_OPTIONS
                + _PLAT_BUILD_OPTIONS.get(
                    context.devices[0].platform.name, []))

        import os
        forced_options = os.environ.get("PYOPENCL_BUILD_OPTIONS")
        if forced_options:
            options = options + forced_options.split()

        # {{{ find include path

        include_path = ["."]

        option_idx = 0
        while option_idx < len(options):
            option = options[option_idx].strip()
            if option.startswith("-I") or option.startswith("/I"):
                if len(option) == 2:
                    if option_idx+1 < len(options):
                        include_path.append(options[option_idx+1])
                    option_idx += 2
                else:
                    include_path.append(option[2:].lstrip())
                    option_idx += 1
            else:
                option_idx += 1

        # }}}

        options = [encode_if_necessary(s) for s in options]

        options = [cls._shlex_quote(s) for s in options]

        return b" ".join(options), include_path

    def build(self, options=[], devices=None, cache_dir=None):
        options_bytes, include_path = self._process_build_options(
                self._context, options)

        if cache_dir is None:
            cache_dir = getattr(self._context, 'cache_dir', None)

        import os
        if os.environ.get("PYOPENCL_NO_CACHE") and self._prg is None:
            self._prg = _cl._Program(self._context, self._source)

        if self._prg is not None:
            # uncached

            self._build_and_catch_errors(
                    lambda: self._prg.build(options_bytes, devices),
                    options_bytes=options_bytes)

        else:
            # cached

            from pyopencl.cache import create_built_program_from_source_cached
            self._prg = self._build_and_catch_errors(
                    lambda: create_built_program_from_source_cached(
                        self._context, self._source, options_bytes, devices,
                        cache_dir=cache_dir, include_path=include_path),
                    options_bytes=options_bytes, source=self._source)

            del self._context

        return self

    def _build_and_catch_errors(self, build_func, options_bytes, source=None):
        try:
            return build_func()
        except _cl.RuntimeError as e:
            msg = e.what
            if options_bytes:
                msg = msg + "\n(options: %s)" % options_bytes.decode("utf-8")

            if source is not None:
                from tempfile import NamedTemporaryFile
                srcfile = NamedTemporaryFile(mode="wt", delete=False, suffix=".cl")
                try:
                    srcfile.write(source)
                finally:
                    srcfile.close()

                msg = msg + "\n(source saved as %s)" % srcfile.name

            code = e.code
            routine = e.routine

            err = _cl.RuntimeError(
                    _cl.Error._ErrorRecord(
                        msg=msg,
                        code=code,
                        routine=routine))

        # Python 3.2 outputs the whole list of currently active exceptions
        # This serves to remove one (redundant) level from that nesting.
        raise err

    # }}}

    def compile(self, options=[], devices=None, headers=[]):
        options_bytes, _ = self._process_build_options(self._context, options)

        return self._prg.compile(options_bytes, devices, headers)

    def __eq__(self, other):
        return self._get_prg() == other._get_prg()

    def __ne__(self, other):
        return self._get_prg() == other._get_prg()

    def __hash__(self):
        return hash(self._get_prg())

_add_get_info_attrs(Program, Program.get_info, program_info)


def create_program_with_built_in_kernels(context, devices, kernel_names):
    if not isinstance(kernel_names, str):
        kernel_names = ":".join(kernel_names)

    return Program(_Program.create_with_built_in_kernels(
        context, devices, kernel_names))


def link_program(context, programs, options=[], devices=None):
    options_bytes, _ = Program._process_build_options(context, options)
    return Program(_Program.link(context, programs, options_bytes, devices))

# }}}


# {{{ create_some_context

def create_some_context(interactive=None, answers=None, cache_dir=None):
    import os
    if answers is None:
        if "PYOPENCL_CTX" in os.environ:
            ctx_spec = os.environ["PYOPENCL_CTX"]
            answers = ctx_spec.split(":")

        if "PYOPENCL_TEST" in os.environ:
            from pyopencl.tools import get_test_platforms_and_devices
            for plat, devs in get_test_platforms_and_devices():
                for dev in devs:
                    return Context([dev], cache_dir=cache_dir)

    if answers is not None:
        pre_provided_answers = answers
        answers = answers[:]
    else:
        pre_provided_answers = None

    user_inputs = []

    if interactive is None:
        interactive = True
        try:
            import sys
            if not sys.stdin.isatty():
                interactive = False
        except:
            interactive = False

    def cc_print(s):
        if interactive:
            print(s)

    def get_input(prompt):
        if answers:
            return str(answers.pop(0))
        elif not interactive:
            return ''
        else:
            user_input = input(prompt)
            user_inputs.append(user_input)
            return user_input

    # {{{ pick a platform

    platforms = get_platforms()

    if not platforms:
        raise Error("no platforms found")
    else:
        if not answers:
            cc_print("Choose platform:")
            for i, pf in enumerate(platforms):
                cc_print("[%d] %s" % (i, pf))

        answer = get_input("Choice [0]:")
        if not answer:
            platform = platforms[0]
        else:
            platform = None
            try:
                int_choice = int(answer)
            except ValueError:
                pass
            else:
                if 0 <= int_choice < len(platforms):
                    platform = platforms[int_choice]

            if platform is None:
                answer = answer.lower()
                for i, pf in enumerate(platforms):
                    if answer in pf.name.lower():
                        platform = pf
                if platform is None:
                    raise RuntimeError("input did not match any platform")

    # }}}

    # {{{ pick a device

    devices = platform.get_devices()

    def parse_device(choice):
        try:
            int_choice = int(choice)
        except ValueError:
            pass
        else:
            if 0 <= int_choice < len(devices):
                return devices[int_choice]

        choice = choice.lower()
        for i, dev in enumerate(devices):
            if choice in dev.name.lower():
                return dev
        raise RuntimeError("input did not match any device")

    if not devices:
        raise Error("no devices found")
    elif len(devices) == 1:
        pass
    else:
        if not answers:
            cc_print("Choose device(s):")
            for i, dev in enumerate(devices):
                cc_print("[%d] %s" % (i, dev))

        answer = get_input("Choice, comma-separated [0]:")
        if not answer:
            devices = [devices[0]]
        else:
            devices = [parse_device(i) for i in answer.split(",")]

    # }}}

    if user_inputs:
        if pre_provided_answers is not None:
            user_inputs = pre_provided_answers + user_inputs
        cc_print("Set the environment variable PYOPENCL_CTX='%s' to "
                "avoid being asked again." % ":".join(user_inputs))

    if answers:
        raise RuntimeError("not all provided choices were used by "
                "create_some_context. (left over: '%s')" % ":".join(answers))

    return Context(devices, cache_dir=cache_dir)

_csc = create_some_context

# }}}


# {{{ enqueue_copy

def _mark_copy_deprecated(func):
    def new_func(*args, **kwargs):
        from warnings import warn
        warn("'%s' has been deprecated in version 2011.1. Please use "
                "enqueue_copy() instead." % func.__name__[1:], DeprecationWarning,
                stacklevel=2)
        return func(*args, **kwargs)

    try:
        from functools import update_wrapper
    except ImportError:
        pass
    else:
        try:
            update_wrapper(new_func, func)
        except AttributeError:
            pass

    return new_func


enqueue_read_image = _mark_copy_deprecated(_cl._enqueue_read_image)
enqueue_write_image = _mark_copy_deprecated(_cl._enqueue_write_image)
enqueue_copy_image = _mark_copy_deprecated(_cl._enqueue_copy_image)
enqueue_copy_image_to_buffer = _mark_copy_deprecated(
        _cl._enqueue_copy_image_to_buffer)
enqueue_copy_buffer_to_image = _mark_copy_deprecated(
        _cl._enqueue_copy_buffer_to_image)
enqueue_read_buffer = _mark_copy_deprecated(_cl._enqueue_read_buffer)
enqueue_write_buffer = _mark_copy_deprecated(_cl._enqueue_write_buffer)
enqueue_copy_buffer = _mark_copy_deprecated(_cl._enqueue_copy_buffer)


if _cl.get_cl_header_version() >= (1, 1):
    enqueue_read_buffer_rect = _mark_copy_deprecated(_cl._enqueue_read_buffer_rect)
    enqueue_write_buffer_rect = _mark_copy_deprecated(_cl._enqueue_write_buffer_rect)
    enqueue_copy_buffer_rect = _mark_copy_deprecated(_cl._enqueue_copy_buffer_rect)


def enqueue_copy(queue, dest, src, **kwargs):
    """Copy from :class:`Image`, :class:`Buffer` or the host to
    :class:`Image`, :class:`Buffer` or the host. (Note: host-to-host
    copies are unsupported.)

    The following keyword arguments are available:

    :arg wait_for: (optional, default empty)
    :arg is_blocking: Wait for completion. Defaults to *True*.
      (Available on any copy involving host memory)

    :return: A :class:`NannyEvent` if the transfer involved a
        host-side buffer, otherwise an :class:`Event`.

    .. note::

        Two types of 'buffer' occur in the arguments to this function,
        :class:`Buffer` and 'host-side buffers'. The latter are
        defined by Python and commonly called `buffer objects
        <https://docs.python.org/3.4/c-api/buffer.html>`_. :mod:`numpy`
        arrays are a very common example.
        Make sure to always be clear on whether a :class:`Buffer` or a
        Python buffer object is needed.

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Buffer` ↔ host
    .. ------------------------------------------------------------------------

    :arg device_offset: offset in bytes (optional)

    .. note::

        The size of the transfer is controlled by the size of the
        of the host-side buffer. If the host-side buffer
        is a :class:`numpy.ndarray`, you can control the transfer size by
        transfering into a smaller 'view' of the target array, like this::

            cl.enqueue_copy(queue, large_dest_numpy_array[:15], src_buffer)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Buffer` ↔ :class:`Buffer`
    .. ------------------------------------------------------------------------

    :arg byte_count: (optional) If not specified, defaults to the
        size of the source in versions 2012.x and earlier,
        and to the minimum of the size of the source and target
        from 2013.1 on.
    :arg src_offset: (optional)
    :arg dest_offset: (optional)

    .. ------------------------------------------------------------------------
    .. rubric :: Rectangular :class:`Buffer` ↔  host transfers (CL 1.1 and newer)
    .. ------------------------------------------------------------------------

    :arg buffer_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg host_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg buffer_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)
    :arg host_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)

    .. ------------------------------------------------------------------------
    .. rubric :: Rectangular :class:`Buffer` ↔  :class:`Buffer`
        transfers (CL 1.1 and newer)
    .. ------------------------------------------------------------------------

    :arg src_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg dst_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg src_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)
    :arg dst_pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional, "tightly-packed" if unspecified)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Image` ↔ host
    .. ------------------------------------------------------------------------

    :arg origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg pitches: :class:`tuple` of :class:`int` of length
        two or shorter. (optional)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Buffer` ↔ :class:`Image`
    .. ------------------------------------------------------------------------

    :arg offset: offset in buffer (mandatory)
    :arg origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`Image` ↔ :class:`Image`
    .. ------------------------------------------------------------------------

    :arg src_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg dest_origin: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)
    :arg region: :class:`tuple` of :class:`int` of length
        three or shorter. (mandatory)

    .. ------------------------------------------------------------------------
    .. rubric :: Transfer :class:`SVM`/host ↔ :class:`SVM`/host
    .. ------------------------------------------------------------------------

    :arg byte_count: (optional) If not specified, defaults to the
        size of the source in versions 2012.x and earlier,
        and to the minimum of the size of the source and target
        from 2013.1 on.

    |std-enqueue-blurb|

    .. versionadded:: 2011.1
    """

    if isinstance(dest, MemoryObjectHolder):
        if dest.type == mem_object_type.BUFFER:
            if isinstance(src, MemoryObjectHolder):
                if src.type == mem_object_type.BUFFER:
                    if "src_origin" in kwargs:
                        return _cl._enqueue_copy_buffer_rect(
                                queue, src, dest, **kwargs)
                    else:
                        kwargs["dst_offset"] = kwargs.pop("dest_offset", 0)
                        return _cl._enqueue_copy_buffer(queue, src, dest, **kwargs)
                elif src.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
                    return _cl._enqueue_copy_image_to_buffer(
                            queue, src, dest, **kwargs)
                else:
                    raise ValueError("invalid src mem object type")
            else:
                # assume from-host
                if "buffer_origin" in kwargs:
                    return _cl._enqueue_write_buffer_rect(queue, dest, src, **kwargs)
                else:
                    return _cl._enqueue_write_buffer(queue, dest, src, **kwargs)

        elif dest.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
            if isinstance(src, MemoryObjectHolder):
                if src.type == mem_object_type.BUFFER:
                    return _cl._enqueue_copy_buffer_to_image(
                            queue, src, dest, **kwargs)
                elif src.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
                    return _cl._enqueue_copy_image(queue, src, dest, **kwargs)
                else:
                    raise ValueError("invalid src mem object type")
            else:
                # assume from-host
                origin = kwargs.pop("origin")
                region = kwargs.pop("region")

                pitches = kwargs.pop("pitches", (0, 0))
                if len(pitches) == 1:
                    kwargs["row_pitch"], = pitches
                else:
                    kwargs["row_pitch"], kwargs["slice_pitch"] = pitches

                return _cl._enqueue_write_image(
                        queue, dest, origin, region, src, **kwargs)
        else:
            raise ValueError("invalid dest mem object type")

    elif isinstance(dest, SVM):
        # to SVM
        if isinstance(src, SVM):
            src = src.mem

        return _cl._enqueue_svm_memcpy(queue, dest.mem, src, **kwargs)
    else:
        # assume to-host

        if isinstance(src, MemoryObjectHolder):
            if src.type == mem_object_type.BUFFER:
                if "buffer_origin" in kwargs:
                    return _cl._enqueue_read_buffer_rect(queue, src, dest, **kwargs)
                else:
                    return _cl._enqueue_read_buffer(queue, src, dest, **kwargs)
            elif src.type in [mem_object_type.IMAGE2D, mem_object_type.IMAGE3D]:
                origin = kwargs.pop("origin")
                region = kwargs.pop("region")

                pitches = kwargs.pop("pitches", (0, 0))
                if len(pitches) == 1:
                    kwargs["row_pitch"], = pitches
                else:
                    kwargs["row_pitch"], kwargs["slice_pitch"] = pitches

                return _cl._enqueue_read_image(
                        queue, src, origin, region, dest, **kwargs)
            else:
                raise ValueError("invalid src mem object type")
        elif isinstance(src, SVM):
            # from svm
            # dest is not a SVM instance, otherwise we'd be in the branch above
            return _cl._enqueue_svm_memcpy(queue, dest, src.mem, **kwargs)
        else:
            # assume from-host
            raise TypeError("enqueue_copy cannot perform host-to-host transfers")

# }}}


# {{{ image creation

DTYPE_TO_CHANNEL_TYPE = {
    np.dtype(np.float32): channel_type.FLOAT,
    np.dtype(np.int16): channel_type.SIGNED_INT16,
    np.dtype(np.int32): channel_type.SIGNED_INT32,
    np.dtype(np.int8): channel_type.SIGNED_INT8,
    np.dtype(np.uint16): channel_type.UNSIGNED_INT16,
    np.dtype(np.uint32): channel_type.UNSIGNED_INT32,
    np.dtype(np.uint8): channel_type.UNSIGNED_INT8,
    }
try:
    np.float16
except:
    pass
else:
    DTYPE_TO_CHANNEL_TYPE[np.dtype(np.float16)] = channel_type.HALF_FLOAT

DTYPE_TO_CHANNEL_TYPE_NORM = {
    np.dtype(np.int16): channel_type.SNORM_INT16,
    np.dtype(np.int8): channel_type.SNORM_INT8,
    np.dtype(np.uint16): channel_type.UNORM_INT16,
    np.dtype(np.uint8): channel_type.UNORM_INT8,
    }


def image_from_array(ctx, ary, num_channels=None, mode="r", norm_int=False):
    if not ary.flags.c_contiguous:
        raise ValueError("array must be C-contiguous")

    dtype = ary.dtype
    if num_channels is None:

        from pyopencl.array import vec
        try:
            dtype, num_channels = vec.type_to_scalar_and_count[dtype]
        except KeyError:
            # It must be a scalar type then.
            num_channels = 1

        shape = ary.shape
        strides = ary.strides

    elif num_channels == 1:
        shape = ary.shape
        strides = ary.strides
    else:
        if ary.shape[-1] != num_channels:
            raise RuntimeError("last dimension must be equal to number of channels")

        shape = ary.shape[:-1]
        strides = ary.strides[:-1]

    if mode == "r":
        mode_flags = mem_flags.READ_ONLY
    elif mode == "w":
        mode_flags = mem_flags.WRITE_ONLY
    else:
        raise ValueError("invalid value '%s' for 'mode'" % mode)

    img_format = {
            1: channel_order.R,
            2: channel_order.RG,
            3: channel_order.RGB,
            4: channel_order.RGBA,
            }[num_channels]

    assert ary.strides[-1] == ary.dtype.itemsize

    if norm_int:
        channel_type = DTYPE_TO_CHANNEL_TYPE_NORM[dtype]
    else:
        channel_type = DTYPE_TO_CHANNEL_TYPE[dtype]

    return Image(ctx, mode_flags | mem_flags.COPY_HOST_PTR,
            ImageFormat(img_format, channel_type),
            shape=shape[::-1], pitches=strides[::-1][1:],
            hostbuf=ary)

# }}}


# {{{ enqueue_* compatibility shims

def enqueue_marker(queue, wait_for=None):
    if queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
        return _cl._enqueue_marker_with_wait_list(queue, wait_for)
    else:
        if wait_for:
            _cl._enqueue_wait_for_events(queue, wait_for)
        return _cl._enqueue_marker(queue)


def enqueue_barrier(queue, wait_for=None):
    if queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2):
        return _cl._enqueue_barrier_with_wait_list(queue, wait_for)
    else:
        _cl._enqueue_barrier(queue)
        if wait_for:
            _cl._enqueue_wait_for_events(queue, wait_for)
        return _cl._enqueue_marker(queue)


def enqueue_fill_buffer(queue, mem, pattern, offset, size, wait_for=None):
    if not (queue._get_cl_version() >= (1, 2) and get_cl_header_version() >= (1, 2)):
        from warnings import warn
        warn("The context for this queue does not declare OpenCL 1.2 support, so "
                "the next thing you might see is a crash")
    return _cl._enqueue_fill_buffer(queue, mem, pattern, offset, size, wait_for)

# }}}


# {{{ numpy-like svm allocation

def svm_empty(ctx, flags, shape, dtype, order="C", alignment=None):
    """Allocate an empty :class:`numpy.ndarray` of the given *shape*, *dtype*
    and *order*. (See :func:`numpy.empty` for the meaning of these arguments.)
    The array will be allocated in shared virtual memory belonging
    to *ctx*.

    :arg ctx: a :class:`Context`
    :arg flags: a combination of flags from :class:`svm_mem_flags`.
    :arg alignment: the number of bytes to which the beginning of the memory
        is aligned. Defaults to the :attr:`numpy.dtype.itemsize` of *dtype*.

    :returns: a :class:`numpy.ndarray` whose :attr:`numpy.ndarray.base` attribute
        is a :class:`SVMAllocation`.

    To pass the resulting array to an OpenCL kernel or :func:`enqueue_copy`, you
    will likely want to wrap the returned array in an :class:`SVM` tag.

    .. versionadded:: 2016.2
    """

    dtype = np.dtype(dtype)

    try:
        s = 1
        for dim in shape:
            s *= dim
    except TypeError:
        import sys
        if sys.version_info >= (3,):
            admissible_types = (int, np.integer)
        else:
            admissible_types = (np.integer,) + six.integer_types

        if not isinstance(shape, admissible_types):
            raise TypeError("shape must either be iterable or "
                    "castable to an integer")
        s = shape
        shape = (shape,)

    itemsize = dtype.itemsize
    nbytes = s * itemsize

    from pyopencl.compyte.array import c_contiguous_strides, f_contiguous_strides

    if order in "fF":
        strides = f_contiguous_strides(itemsize, shape)
    elif order in "cC":
        strides = c_contiguous_strides(itemsize, shape)
    else:
        raise ValueError("order not recognized: %s" % order)

    descr = dtype.descr

    interface = {
        "version": 3,
        "shape": shape,
        "strides": strides,
        }

    if len(descr) == 1:
        interface["typestr"] = descr[0][1]
    else:
        interface["typestr"] = "V%d" % itemsize
        interface["descr"] = descr

    if alignment is None:
        alignment = itemsize

    svm_alloc = SVMAllocation(ctx, nbytes, alignment, flags, _interface=interface)
    return np.asarray(svm_alloc)


def svm_empty_like(ctx, flags, ary, alignment=None):
    """Allocate an empty :class:`numpy.ndarray` like the existing
    :class:`numpy.ndarray` *ary*.  The array will be allocated in shared
    virtual memory belonging to *ctx*.

    :arg ctx: a :class:`Context`
    :arg flags: a combination of flags from :class:`svm_mem_flags`.
    :arg alignment: the number of bytes to which the beginning of the memory
        is aligned. Defaults to the :attr:`numpy.dtype.itemsize` of *dtype*.

    :returns: a :class:`numpy.ndarray` whose :attr:`numpy.ndarray.base` attribute
        is a :class:`SVMAllocation`.

    To pass the resulting array to an OpenCL kernel or :func:`enqueue_copy`, you
    will likely want to wrap the returned array in an :class:`SVM` tag.

    .. versionadded:: 2016.2
    """
    if ary.flags.c_contiguous:
        order = "C"
    elif ary.flags.f_contiguous:
        order = "F"
    else:
        raise ValueError("array is neither C- nor Fortran-contiguous")

    return svm_empty(ctx, ary.shape, ary.dtype, order,
            flags=flags, alignment=alignment)


def csvm_empty(ctx, shape, dtype, order="C", alignment=None):
    """
    Like :func:`svm_empty`, but with *flags* set for a coarse-grain read-write
    buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty(ctx, svm_mem_flags.READ_WRITE, shape, dtype, order, alignment)


def csvm_empty_like(ctx, ary, alignment=None):
    """
    Like :func:`svm_empty_like`, but with *flags* set for a coarse-grain
    read-write buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty_like(ctx, svm_mem_flags.READ_WRITE, ary)


def fsvm_empty(ctx, shape, dtype, order="C", alignment=None):
    """
    Like :func:`svm_empty`, but with *flags* set for a fine-grain read-write
    buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty(ctx,
            svm_mem_flags.READ_WRITE | svm_mem_flags.SVM_FINE_GRAIN_BUFFER,
            shape, dtype, order, alignment)


def fsvm_empty_like(ctx, ary, alignment=None):
    """
    Like :func:`svm_empty_like`, but with *flags* set for a fine-grain
    read-write buffer.

    .. versionadded:: 2016.2
    """
    return svm_empty_like(
            ctx,
            svm_mem_flags.READ_WRITE | svm_mem_flags.SVM_FINE_GRAIN_BUFFER,
            ary)

# }}}

# vim: foldmethod=marker
