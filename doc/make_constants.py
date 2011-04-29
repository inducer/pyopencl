import pyopencl as cl

devi = cl.device_info
ctxi = cl.context_info
ctxp = cl.context_properties
fpc = cl.device_fp_config
cho = cl.channel_order
wgi = cl.kernel_work_group_info
iam = cl.addressing_mode
evi = cl.event_info
memi = cl.mem_info
ctype = cl.command_type
memf = cl.mem_flags
dppe = cl.device_partition_property_ext
ade = cl.affinity_domain_ext

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
        devi: {
            getattr(devi, "PLATFORM_NOT_FOUND_KHR", None):
            ("cl_khr_icd", "2011.1"),

            getattr(devi, "PREFERRED_VECTOR_WIDTH_HALF", None): ("CL_1.1", "0.92"),
            getattr(devi, "HOST_UNIFIED_MEMORY", None): ("CL_1.1", "0.92"),
            getattr(devi, "NATIVE_VECTOR_WIDTH_CHAR", None): ("CL_1.1", "0.92"),
            getattr(devi, "NATIVE_VECTOR_WIDTH_SHORT", None): ("CL_1.1", "0.92"),
            getattr(devi, "NATIVE_VECTOR_WIDTH_INT", None): ("CL_1.1", "0.92"),
            getattr(devi, "NATIVE_VECTOR_WIDTH_LONG", None): ("CL_1.1", "0.92"),
            getattr(devi, "NATIVE_VECTOR_WIDTH_FLOAT", None): ("CL_1.1", "0.92"),
            getattr(devi, "NATIVE_VECTOR_WIDTH_DOUBLE", None): ("CL_1.1", "0.92"),
            getattr(devi, "NATIVE_VECTOR_WIDTH_HALF", None): ("CL_1.1", "0.92"),
            getattr(devi, "OPENCL_C_VERSION", None): ("CL_1.1", "0.92"),
            getattr(devi, "COMPUTE_CAPABILITY_MAJOR_NV", None):
            ("cl_nv_device_attribute_query", "0.92"),
            getattr(devi, "COMPUTE_CAPABILITY_MINOR_NV", None):
            ("cl_nv_device_attribute_query", "0.92"),
            getattr(devi, "REGISTERS_PER_BLOCK_NV", None):
            ("cl_nv_device_attribute_query", "0.92"),
            getattr(devi, "WARP_SIZE_NV", None):
            ("cl_nv_device_attribute_query", "0.92"),
            getattr(devi, "GPU_OVERLAP_NV", None):
            ("cl_nv_device_attribute_query", "0.92"),
            getattr(devi, "KERNEL_EXEC_TIMEOUT_NV", None):
            ("cl_nv_device_attribute_query", "0.92"),
            getattr(devi, "INTEGRATED_MEMORY_NV", None):
            ("cl_nv_device_attribute_query", "0.92"),

            getattr(devi, "DOUBLE_FP_CONFIG", None):
            ("cl_khr_fp64", "2011.1"),
            getattr(devi, "HALF_FP_CONFIG", None):
            ("cl_khr_fp16", "2011.1"),

            getattr(devi, "PROFILING_TIMER_OFFSET_AMD", None):
            ("cl_amd_device_attribute_query", "2011.1"),

            getattr(devi, "PARENT_DEVICE_EXT", None):
            ("cl_ext_device_fission", "2011.1"),
            getattr(devi, "PARTITION_TYPES_EXT", None):
            ("cl_ext_device_fission", "2011.1"),
            getattr(devi, "AFFINITY_DOMAINS_EXT", None):
            ("cl_ext_device_fission", "2011.1"),
            getattr(devi, "REFERENCE_COUNT_EXT", None):
            ("cl_ext_device_fission", "2011.1"),
            getattr(devi, "PARTITION_STYLE_EXT", None):
            ("cl_ext_device_fission", "2011.1"),
            },

        ctxp: {
            getattr(ctxp, "GL_CONTEXT_KHR", None): ("cl_khr_gl_sharing", "0.92"),
            getattr(ctxp, "EGL_DISPLAY_KHR", None): ("cl_khr_gl_sharing", "0.92"),
            getattr(ctxp, "GLX_DISPLAY_KHR", None): ("cl_khr_gl_sharing", "0.92"),
            getattr(ctxp, "WGL_HDC_KHR", None): ("cl_khr_gl_sharing", "0.92"),
            getattr(ctxp, "CGL_SHAREGROUP_KHR", None): ("cl_khr_gl_sharing", "0.92"),

            getattr(ctxp, "OFFLINE_DEVICES_AMD", None): 
            ("cl_amd_offline_devices", "2011.1"),
            },

        fpc: {
            getattr(fpc, "SOFT_FLOAT", None): ("CL_1.1", "0.92"),
            },

        ctxi: {
            getattr(ctxi, "NUM_DEVICES", None): ("CL_1.1", "0.92"),
            },

        cho: {
            getattr(cho, "Rx", None): ("CL_1.1", "0.92"),
            getattr(cho, "RGx", None): ("CL_1.1", "0.92"),
            getattr(cho, "RGBx", None): ("CL_1.1", "0.92"),
            },

        wgi: {
            getattr(wgi, "PREFERRED_WORK_GROUP_SIZE_MULTIPLE", None): ("CL_1.1", "0.92"),
            getattr(wgi, "PRIVATE_MEM_SIZE", None): ("CL_1.1", "0.92"),
            },

        iam: {
            getattr(iam, "MIRRORED_REPEAT", None): ("CL_1.1", "0.92"),
            },

        evi: {
            getattr(evi, "CONTEXT", None): ("CL_1.1", "0.92"),
            },

        memi: {
            getattr(memi, "ASSOCIATED_MEMOBJECT", None): ("CL_1.1", "0.92"),
            getattr(memi, "OFFSET", None): ("CL_1.1", "0.92"),
            },

        ctype: {
            getattr(ctype, "READ_BUFFER_RECT", None): ("CL_1.1", "0.92"),
            getattr(ctype, "WRITE_BUFFER_RECT", None): ("CL_1.1", "0.92"),
            getattr(ctype, "COPY_BUFFER_RECT", None): ("CL_1.1", "0.92"),
            getattr(ctype, "USER", None): ("CL_1.1", "0.92"),
            },

        memf: {
            getattr(memf, "USE_PERSISTENT_MEM_AMD", None): 
            ("cl_amd_device_memory_flags", "2011.1"),
            },
        dppe: {
            getattr(dppe, "EQUALLY", None): ("cl_ext_device_fission", "2011.1"),
            getattr(dppe, "BY_COUNTS", None): ("cl_ext_device_fission", "2011.1"),
            getattr(dppe, "BY_NAMES", None): ("cl_ext_device_fission", "2011.1"),
            getattr(dppe, "BY_AFFINITY_DOMAIN", None): ("cl_ext_device_fission", "2011.1"),

            getattr(dppe, "PROPERTIES_LIST_END", None): ("cl_ext_device_fission", "2011.1"),
            getattr(dppe, "PARTITION_BY_COUNTS_LIST_END", None): ("cl_ext_device_fission", "2011.1"),
            getattr(dppe, "PARTITION_BY_NAMES_LIST_END", None): ("cl_ext_device_fission", "2011.1"),
            },
        ade: {
            getattr(ade, "L1_CACHE", None): ("cl_ext_device_fission", "2011.1"),
            getattr(ade, "L2_CACHE", None): ("cl_ext_device_fission", "2011.1"),
            getattr(ade, "L3_CACHE", None): ("cl_ext_device_fission", "2011.1"),
            getattr(ade, "L4_CACHE", None): ("cl_ext_device_fission", "2011.1"),
            getattr(ade, "NUMA", None): ("cl_ext_device_fission", "2011.1"),
            getattr(ade, "NEXT_FISSIONABLE", None): ("cl_ext_device_fission", "2011.1"),
            }
        }
try:
    gl_ci = cl.gl_context_info
except AttributeError:
    pass
else:
    const_ext_lookup[gl_ci] = {
            getattr(gl_ci, "CURRENT_DEVICE_FOR_GL_CONTEXT_KHR", None):
            ("cl_khr_gl_sharing", "0.92"),

            getattr(gl_ci, "DEVICES_FOR_GL_CONTEXT_KHR", None):
            ("cl_khr_gl_sharing", "0.92"),
            }

cls_ext_lookup = {
        #cl.buffer_create_type: ("CL_1.1", "0.92"),
        }


def doc_class(cls):
    print ".. class :: %s" % cls.__name__
    print
    if cls.__name__.startswith("gl_"):
        print "    Only available when PyOpenCL is compiled with GL support. See :func:`have_gl`."
        print

    if cls in cls_ext_lookup:
        for l in get_extra_lines(cls_ext_lookup[cls]):
            print l

    cls_const_ext = const_ext_lookup.get(cls, {})
    for i in sorted(dir(cls)):
        if not i.startswith("_")  and not i == "to_string":
            print "    .. attribute :: %s" % i
            value = getattr(cls, i)

            if value in cls_const_ext:
                for l in get_extra_lines(cls_const_ext[value]):
                    print "    "+l

    print "    .. method :: to_string(value)"
    print
    print "        Returns a :class:`str` representing *value*."
    print
    print "        .. versionadded:: 0.91"
    print


if not cl.have_gl():
    print ".. warning::"
    print
    print "    This set of PyOpenCL documentation is incomplete because it"
    print "    was generated on a PyOpenCL build that did not support OpenGL."
    print

print ".. This is an automatically generated file. DO NOT EDIT"
print
for cls in cl.CONSTANT_CLASSES:
    doc_class(cls)
