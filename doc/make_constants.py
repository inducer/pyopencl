from __future__ import absolute_import, print_function

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

fission = ("cl_ext_device_fission", "2011.1")
nv_devattr = ("cl_nv_device_attribute_query", "0.92")
gl_sharing = ("cl_khr_gl_sharing", "0.92")
cl_spir_devattr = ("cl_khr_spir", "2016.2")
cl_11 = ("CL_1.1", "0.92")
cl_12 = ("CL_1.2", "2011.2")
cl_12_2015 = ("CL_1.2", "2015.2")
cl_20 = ("CL_2.0", "2015.2")
cl_21 = ("CL_2.1", "2016.2")
amd_devattr = ("cl_amd_device_attribute_query", "2013.2")
qcom_hp_devattr = ("cl_qcom_ext_host_ptr", "2016.2")
intel_me_devattr = ("cl_intel_advanced_motion_estimation", "2016.2")
intel_ss_devattr = ("cl_intel_simultaneous_sharing", "2016.2")
altera_temp_devattr = ("cl_altera_device_temperature", "2016.2")


def get_extra_lines(tup):
    ext_name, pyopencl_ver = tup
    if ext_name is not None:
        if ext_name.startswith("CL_"):
            # capital letters -> CL version, not extension
            yield ""
            yield "    Available with OpenCL %s." % (
                    ext_name[3:])
            yield ""

        else:
            yield ""
            yield "    Available with the ``%s`` extension." % ext_name
            yield ""

    if pyopencl_ver is not None:
        yield ""
        yield "    .. versionadded:: %s" % pyopencl_ver
        yield ""

const_ext_lookup = {
        cl.status_code: {
            "PLATFORM_NOT_FOUND_KHR": ("cl_khr_icd", "2011.1"),

            "INVALID_GL_SHAREGROUP_REFERENCE_KHR": gl_sharing,

            "MISALIGNED_SUB_BUFFER_OFFSET": cl_11,
            "EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST": cl_11,
            "INVALID_GLOBAL_WORK_SIZE": cl_11,

            "COMPILE_PROGRAM_FAILURE": cl_12,
            "LINKER_NOT_AVAILABLE": cl_12,
            "LINK_PROGRAM_FAILURE": cl_12,
            "DEVICE_PARTITION_FAILED": cl_12,
            "KERNEL_ARG_INFO_NOT_AVAILABLE": cl_12,
            "INVALID_IMAGE_DESCRIPTOR": cl_12,
            "INVALID_COMPILER_OPTIONS": cl_12,
            "INVALID_LINKER_OPTIONS": cl_12,
            "INVALID_DEVICE_PARTITION_COUNT": cl_12,

            "INVALID_PIPE_SIZE": cl_20,
            "INVALID_DEVICE_QUEUE": cl_20,

            },

        cl.device_info: {
            "PREFERRED_VECTOR_WIDTH_HALF": cl_11,
            "HOST_UNIFIED_MEMORY": cl_11,
            "NATIVE_VECTOR_WIDTH_CHAR": cl_11,
            "NATIVE_VECTOR_WIDTH_SHORT": cl_11,
            "NATIVE_VECTOR_WIDTH_INT": cl_11,
            "NATIVE_VECTOR_WIDTH_LONG": cl_11,
            "NATIVE_VECTOR_WIDTH_FLOAT": cl_11,
            "NATIVE_VECTOR_WIDTH_DOUBLE": cl_11,
            "NATIVE_VECTOR_WIDTH_HALF": cl_11,
            "OPENCL_C_VERSION": cl_11,
            "SPIR_VERSIONS": cl_spir_devattr,
            "COMPUTE_CAPABILITY_MAJOR_NV": nv_devattr,
            "COMPUTE_CAPABILITY_MINOR_NV": nv_devattr,
            "REGISTERS_PER_BLOCK_NV": nv_devattr,
            "WARP_SIZE_NV": nv_devattr,
            "GPU_OVERLAP_NV": nv_devattr,
            "KERNEL_EXEC_TIMEOUT_NV": nv_devattr,
            "INTEGRATED_MEMORY_NV": nv_devattr,
            "ATTRIBUTE_ASYNC_ENGINE_COUNT_NV": nv_devattr,
            "PCI_BUS_ID_NV": nv_devattr,
            "PCI_BUS_SLOT_NV": nv_devattr,

            "DOUBLE_FP_CONFIG":
            ("cl_khr_fp64", "2011.1"),
            "HALF_FP_CONFIG":
            ("cl_khr_fp16", "2011.1"),

            "PROFILING_TIMER_OFFSET_AMD": amd_devattr,
            "TOPOLOGY_AMD": amd_devattr,
            "BOARD_NAME_AMD": amd_devattr,
            "GLOBAL_FREE_MEMORY_AMD": amd_devattr,
            "SIMD_PER_COMPUTE_UNIT_AMD": amd_devattr,
            "SIMD_WIDTH_AMD": amd_devattr,
            "SIMD_INSTRUCTION_WIDTH_AMD": amd_devattr,
            "WAVEFRONT_WIDTH_AMD": amd_devattr,
            "GLOBAL_MEM_CHANNELS_AMD": amd_devattr,
            "GLOBAL_MEM_CHANNEL_BANKS_AMD": amd_devattr,
            "GLOBAL_MEM_CHANNEL_BANK_WIDTH_AMD": amd_devattr,
            "LOCAL_MEM_SIZE_PER_COMPUTE_UNIT_AMD": amd_devattr,
            "LOCAL_MEM_BANKS_AMD": amd_devattr,
            "THREAD_TRACE_SUPPORTED_AMD": amd_devattr,
            "GFXIP_MAJOR_AMD": amd_devattr,
            "GFXIP_MINOR_AMD": amd_devattr,
            "AVAILABLE_ASYNC_QUEUES_AMD": amd_devattr,

            "ME_VERSION_INTEL": intel_me_devattr,
            "SIMULTANEOUS_INTEROPS_INTEL": intel_ss_devattr,
            "NUM_SIMULTANEOUS_INTEROPS_INTEL": intel_ss_devattr,

            "EXT_MEM_PADDING_IN_BYTES_QCOM": qcom_hp_devattr,
            "PAGE_SIZE_QCOM": qcom_hp_devattr,

            "CORE_TEMPERATURE_ALTERA": altera_temp_devattr,

            "MAX_ATOMIC_COUNTERS_EXT":
            ("cl_ext_atomic_counters_64", "2013.2"),

            "PARENT_DEVICE_EXT":
            fission,
            "PARTITION_TYPES_EXT":
            fission,
            "AFFINITY_DOMAINS_EXT":
            fission,
            "REFERENCE_COUNT_EXT":
            fission,
            "PARTITION_STYLE_EXT": fission,

            "LINKER_AVAILABLE": cl_12,
            "BUILT_IN_KERNELS": cl_12,
            "IMAGE_MAX_BUFFER_SIZE": cl_12,
            "IMAGE_MAX_ARRAY_SIZE": cl_12,
            "PARENT_DEVICE": cl_12,
            "PARTITION_MAX_SUB_DEVICES": cl_12,
            "PARTITION_PROPERTIES": cl_12,
            "PARTITION_AFFINITY_DOMAIN": cl_12,
            "PARTITION_TYPE": cl_12,
            "REFERENCE_COUNT": cl_12,
            "PREFERRED_INTEROP_USER_SYNC": cl_12,
            "PRINTF_BUFFER_SIZE": cl_12,

            "DEVICE_ON_HOST_PROPERTIES": cl_20,

            "MAX_READ_WRITE_IMAGE_ARGS": cl_20,
            "MAX_GLOBAL_VARIABLE_SIZE": cl_20,
            "QUEUE_ON_DEVICE_PROPERTIES": cl_20,
            "QUEUE_ON_DEVICE_PREFERRED_SIZE": cl_20,
            "QUEUE_ON_DEVICE_MAX_SIZE": cl_20,
            "MAX_ON_DEVICE_QUEUES": cl_20,
            "MAX_ON_DEVICE_EVENTS": cl_20,
            "SVM_CAPABILITIES": cl_20,
            "GLOBAL_VARIABLE_PREFERRED_TOTAL_SIZE": cl_20,
            "MAX_PIPE_ARGS": cl_20,
            "PIPE_MAX_ACTIVE_RESERVATIONS": cl_20,
            "PIPE_MAX_PACKET_SIZE": cl_20,
            "PREFERRED_PLATFORM_ATOMIC_ALIGNMENT": cl_20,
            "PREFERRED_GLOBAL_ATOMIC_ALIGNMENT": cl_20,
            "PREFERRED_LOCAL_ATOMIC_ALIGNMENT": cl_20,
            "IL_VERSION": cl_21,
            "MAX_NUM_SUB_GROUPS": cl_21,
            "SUB_GROUP_INDEPENDENT_FORWARD_PROGRESS": cl_21,
            },

        cl.mem_object_type: {
            "IMAGE2D_ARRAY": cl_12,
            "IMAGE1D": cl_12,
            "IMAGE1D_ARRAY": cl_12,
            "IMAGE1D_BUFFER": cl_12,

            "PIPE": cl_20,
            },

        cl.device_type: {
            "CUSTOM": cl_12,
            },

        cl.context_properties: {
            "GL_CONTEXT_KHR": gl_sharing,
            "EGL_DISPLAY_KHR": gl_sharing,
            "GLX_DISPLAY_KHR": gl_sharing,
            "WGL_HDC_KHR": gl_sharing,
            "CGL_SHAREGROUP_KHR": gl_sharing,

            "OFFLINE_DEVICES_AMD":
            ("cl_amd_offline_devices", "2011.1"),
            },

        cl.device_fp_config: {
            "SOFT_FLOAT": cl_11,
            "CORRECTLY_ROUNDED_DIVIDE_SQRT": cl_12,
            },

        cl.command_queue_properties: {
            "ON_DEVICE": cl_20,
            "ON_DEVICE_DEFAULT": cl_20,
            },

        cl.context_info: {
            "NUM_DEVICES": cl_11,
            "INTEROP_USER_SYNC": cl_12,
            },

        cl.channel_order: {
            "Rx": cl_11,
            "RGx": cl_11,
            "RGBx": cl_11,

            "sRGB": cl_20,
            "sRGBx": cl_20,
            "sRGBA": cl_20,
            "sBGRA": cl_20,
            "ABGR": cl_20,
            },

        cl.kernel_work_group_info: {
            "PREFERRED_WORK_GROUP_SIZE_MULTIPLE": cl_11,
            "PRIVATE_MEM_SIZE": cl_11,
            "GLOBAL_WORK_SIZE": cl_12,
            },

        cl.addressing_mode: {
            "MIRRORED_REPEAT": cl_11,
            },

        cl.sampler_info: {
            "MIP_FILTER_MODE": cl_20,
            "LOD_MIN": cl_20,
            "LOD_MAX": cl_20,
            },

        cl.event_info: {
            "CONTEXT": cl_11,
            },

        cl.mem_info: {
            "ASSOCIATED_MEMOBJECT": cl_11,
            "OFFSET": cl_11,

            "USES_SVM_POINTER": cl_20,
            },

        cl.image_info: {
            "ARRAY_SIZE": cl_12,
            "BUFFER": cl_12,
            "NUM_MIP_LEVELS": cl_12,
            "NUM_SAMPLES": cl_12,
            },

        cl.map_flags: {
            "WRITE_INVALIDATE_REGION": cl_12,
            },

        cl.program_info: {
            "NUM_KERNELS": cl_12,
            "KERNEL_NAMES": cl_12,
            },

        cl.program_build_info: {
            "BINARY_TYPE": cl_12,

            "GLOBAL_VARIABLE_TOTAL_SIZE": cl_20,
            },

        cl.program_binary_type: {
            "NONE": cl_12,
            "COMPILED_OBJECT": cl_12,
            "LIBRARY": cl_12,
            "EXECUTABLE": cl_12,
            },

        cl.kernel_info: {
            "ATTRIBUTES": cl_12,
            },

        cl.kernel_arg_info: {
            "ADDRESS_QUALIFIER": cl_12,
            "ACCESS_QUALIFIER": cl_12,
            "TYPE_NAME": cl_12,
            "TYPE_QUALIFIER": cl_12_2015,
            "ARG_NAME": cl_12,
            },

        cl.kernel_arg_address_qualifier: {
            "GLOBAL": cl_12,
            "LOCAL": cl_12,
            "CONSTANT": cl_12,
            "PRIVATE": cl_12,
            },

        cl.kernel_arg_access_qualifier: {
            "READ_ONLY": cl_12,
            "WRITE_ONLY": cl_12,
            "READ_WRITE": cl_12,
            "NONE": cl_12,
            },

        cl.kernel_arg_type_qualifier: {
            "NONE": cl_12_2015,
            "CONST": cl_12_2015,
            "RESTRICT": cl_12_2015,
            "VOLATILE": cl_12_2015,

            "PIPE": cl_20,
            },

        cl.command_type: {
            "READ_BUFFER_RECT": cl_11,
            "WRITE_BUFFER_RECT": cl_11,
            "COPY_BUFFER_RECT": cl_11,
            "USER": cl_11,
            "MIGRATE_MEM_OBJECT_EXT": ("cl_ext_migrate_memobject", "2011.2"),
            "BARRIER": cl_12,
            "MIGRATE_MEM_OBJECTS": cl_12,
            "FILL_BUFFER": cl_12,
            "FILL_IMAGE": cl_12,

            "SVM_FREE": cl_20,
            "SVM_MEMCPY": cl_20,
            "SVM_MEMFILL": cl_20,
            "SVM_MAP": cl_20,
            "SVM_UNMAP": cl_20,
            },

        cl.command_queue_info: {
            "SIZE": cl_20,
            },

        cl.queue_properties: {
            "PROPERTIES": cl_20,
            "SIZE": cl_20,
            },

        cl.mem_flags: {
            "USE_PERSISTENT_MEM_AMD":
            ("cl_amd_device_memory_flags", "2011.1"),
            "HOST_WRITE_ONLY": cl_12,
            "KERNEL_READ_AND_WRITE": cl_20,
            },

        cl.svm_mem_flags: {
            "READ_WRITE": cl_20,
            "WRITE_ONLY": cl_20,
            "READ_ONLY": cl_20,
            "SVM_FINE_GRAIN_BUFFER": cl_20,
            "SVM_ATOMICS": cl_20,
            },

        cl.device_svm_capabilities: {
            "COARSE_GRAIN_BUFFER": cl_20,
            "FINE_GRAIN_BUFFER": cl_20,
            "FINE_GRAIN_SYSTEM": cl_20,
            "ATOMICS": cl_20,
            },

        cl.device_partition_property: {
            "EQUALLY": cl_12,
            "BY_COUNTS": cl_12,
            "BY_NAMES": cl_12,
            "BY_AFFINITY_DOMAIN": cl_12,

            "PROPERTIES_LIST_END": cl_12,
            "PARTITION_BY_COUNTS_LIST_END": cl_12,
            "PARTITION_BY_NAMES_LIST_END": cl_12,
            },

        cl.device_affinity_domain: {
            "NUMA": cl_12,
            "L4_CACHE": cl_12,
            "L3_CACHE": cl_12,
            "L2_CACHE": cl_12,
            "L1_CACHE": cl_12,
            "NEXT_PARITIONNABLE": cl_12,
            },

        cl.profiling_info: {
            "COMPLETE": cl_20,
            },

        cl.mem_migration_flags: {
            "HOST": cl_12,
            "CONTENT_UNDEFINED": cl_12,
            },

        cl.migrate_mem_object_flags_ext: {
            "HOST": ("cl_ext_migrate_memobject", "2011.2"),
            },
        }
try:
    gl_ci = cl.gl_context_info
except AttributeError:
    pass
else:
    const_ext_lookup[gl_ci] = {
            getattr(gl_ci, "CURRENT_DEVICE_FOR_GL_CONTEXT_KHR", None):
            gl_sharing,

            getattr(gl_ci, "DEVICES_FOR_GL_CONTEXT_KHR", None):
            gl_sharing,
            }

cls_ext_lookup = {
        #cl.buffer_create_type: ("CL_1.1", "0.92"),
        }


def doc_class(cls):
    print(".. class :: %s" % cls.__name__)
    print()
    if cls.__name__.startswith("gl_"):
        print("    Only available when PyOpenCL is compiled with GL support.")
        print("    See :func:`have_gl`.")
        print()

    if cls in cls_ext_lookup:
        for l in get_extra_lines(cls_ext_lookup[cls]):
            print(l)

    cls_const_ext = const_ext_lookup.get(cls, {})
    for name in sorted(dir(cls)):
        if not name.startswith("_") and name not in ["to_string", "names", "values"]:
            print("    .. attribute :: %s" % name)

            if name in cls_const_ext:
                for l in get_extra_lines(cls_const_ext[name]):
                    print("    "+l)

    print("    .. method :: to_string(value)")
    print()
    print("        Returns a :class:`str` representing *value*.")
    print()
    print("        .. versionadded:: 0.91")
    print()


if not cl.have_gl():
    print(".. warning::")
    print()
    print("    This set of PyOpenCL documentation is incomplete because it")
    print("    was generated on a PyOpenCL build that did not support OpenGL.")
    print()

import inspect

CONSTANT_CLASSES = [
        getattr(cl, name) for name in dir(cl)
        if inspect.isclass(getattr(cl, name))
        and name[0].islower() and name not in ["zip", "map", "range"]]

print(".. This is an automatically generated file. DO NOT EDIT")
print()
for cls in CONSTANT_CLASSES:
    doc_class(cls)
