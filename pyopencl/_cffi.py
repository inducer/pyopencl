from cffi import FFI

_ffi = FFI()
_cl_header = """

/* gl.h */
typedef unsigned int    GLenum;
typedef int             GLint;          /* 4-byte signed */
typedef unsigned int    GLuint;         /* 4-byte unsigned */


/* cl.h */
/* scalar types */
typedef int8_t          cl_char;
typedef uint8_t         cl_uchar;
typedef int16_t         cl_short;
typedef uint16_t        cl_ushort;
typedef int32_t         cl_int;
typedef uint32_t        cl_uint;
typedef int64_t         cl_long;
typedef uint64_t        cl_ulong;

typedef uint16_t        cl_half;
typedef float                   cl_float;
typedef double                  cl_double;


typedef struct _cl_platform_id *    cl_platform_id;
typedef struct _cl_device_id *      cl_device_id;
typedef struct _cl_context *        cl_context;
typedef struct _cl_command_queue *  cl_command_queue;
typedef struct _cl_mem *            cl_mem;
typedef struct _cl_program *        cl_program;
typedef struct _cl_kernel *         cl_kernel;
typedef struct _cl_event *          cl_event;
typedef struct _cl_sampler *        cl_sampler;

/* WARNING!  Unlike cl_ types in cl_platform.h, cl_bool is not guaranteed to be
the same size as the bool in kernels. */
typedef cl_uint             cl_bool;
typedef cl_ulong            cl_bitfield;
typedef cl_bitfield         cl_device_type;
typedef cl_uint             cl_platform_info;
typedef cl_uint             cl_device_info;
typedef cl_bitfield         cl_device_fp_config;
typedef cl_uint             cl_device_mem_cache_type;
typedef cl_uint             cl_device_local_mem_type;
typedef cl_bitfield         cl_device_exec_capabilities;
typedef cl_bitfield         cl_command_queue_properties;
typedef intptr_t            cl_device_partition_property;
typedef cl_bitfield         cl_device_affinity_domain;

typedef intptr_t            cl_context_properties;
typedef cl_uint             cl_context_info;
typedef cl_uint             cl_command_queue_info;
typedef cl_uint             cl_channel_order;
typedef cl_uint             cl_channel_type;
typedef cl_bitfield         cl_mem_flags;
typedef cl_uint             cl_mem_object_type;
typedef cl_uint             cl_mem_info;
typedef cl_bitfield         cl_mem_migration_flags;
typedef cl_uint             cl_image_info;
typedef cl_uint             cl_buffer_create_type;
typedef cl_uint             cl_addressing_mode;
typedef cl_uint             cl_filter_mode;
typedef cl_uint             cl_sampler_info;
typedef cl_bitfield         cl_map_flags;
typedef cl_uint             cl_program_info;
typedef cl_uint             cl_program_build_info;
typedef cl_uint             cl_program_binary_type;
typedef cl_int              cl_build_status;
typedef cl_uint             cl_kernel_info;
typedef cl_uint             cl_kernel_arg_info;
typedef cl_uint             cl_kernel_arg_address_qualifier;
typedef cl_uint             cl_kernel_arg_access_qualifier;
typedef cl_bitfield         cl_kernel_arg_type_qualifier;
typedef cl_uint             cl_kernel_work_group_info;
typedef cl_uint             cl_event_info;
typedef cl_uint             cl_command_type;
typedef cl_uint             cl_profiling_info;

typedef struct _cl_image_format {
    cl_channel_order        image_channel_order;
    cl_channel_type         image_channel_data_type;
} cl_image_format;

typedef struct _cl_buffer_region {
    size_t                  origin;
    size_t                  size;
} cl_buffer_region;

"""


def _get_wrap_header(filename):
    from pkg_resources import Requirement, resource_filename
    header_name = resource_filename(
            Requirement.parse("pyopencl"), "pyopencl/c_wrapper/"+filename)

    with open(header_name, "rt") as f:
        return f.read()

_ffi.cdef(_cl_header)
_ffi.cdef(_get_wrap_header("wrap_cl_core.h"))


# Copied from pypy distutils/commands/build_ext.py
def _get_c_extension_suffix():
    import imp
    for ext, mod, typ in imp.get_suffixes():
        if typ == imp.C_EXTENSION:
            return ext


def _get_wrapcl_so_names():
    import os.path
    current_directory = os.path.dirname(__file__)

    # TODO: windows debug_mode?

    # Copied from pypy's distutils that "should work for CPython too".
    ext_suffix = _get_c_extension_suffix()
    if ext_suffix is not None:
        yield os.path.join(current_directory, "_wrapcl" + ext_suffix)

        # Oh god. Chop hyphen-separated bits off the end, in the hope that
        # something matches...

        root, ext = os.path.splitext(ext_suffix)
        while True:
            last_hyphen = root.rfind("-")
            if last_hyphen == -1:
                break
            root = root[:last_hyphen]
            yield os.path.join(current_directory, "_wrapcl" + root + ext)

        yield os.path.join(current_directory, "_wrapcl" + ext)

    from distutils.sysconfig import get_config_var
    # "SO" is apparently deprecated, but changing this to "EXT_SUFFIX"
    # as recommended breaks Py2 and PyPy3, as reported by @yuyichao
    # on 2014-07-20.
    #
    # You've been warned. Change "SO" with care.
    yield os.path.join(current_directory, "_wrapcl" + get_config_var("SO"))


def _import_library():
    names = list(_get_wrapcl_so_names())
    for name in names:
        try:
            return _ffi.dlopen(name)
        except OSError:
            pass

    raise RuntimeError("could not find PyOpenCL wrapper library. (tried: %s)"
        % ", ".join(names))

_lib = _import_library()

if _lib.pyopencl_have_gl():
    _ffi.cdef(_get_wrap_header("wrap_cl_gl_core.h"))
