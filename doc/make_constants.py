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
            },

        ctxp: {
            getattr(ctxp, "GL_CONTEXT_KHR", None): ("cl_khr_gl_sharing", "0.92"),
            getattr(ctxp, "EGL_DISPLAY_KHR", None): ("cl_khr_gl_sharing", "0.92"),
            getattr(ctxp, "GLX_DISPLAY_KHR", None): ("cl_khr_gl_sharing", "0.92"),
            getattr(ctxp, "WGL_HDC_KHR", None): ("cl_khr_gl_sharing", "0.92"),
            getattr(ctxp, "CGL_SHAREGROUP_KHR", None): ("cl_khr_gl_sharing", "0.92"),
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
