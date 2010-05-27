import pyopencl as cl

ctxp = cl.context_properties
ext_lookup = {
        ctxp: {
            ctxp.GL_CONTEXT_KHR: "cl_khr_gl_sharing",
            ctxp.EGL_DISPLAY_KHR: "cl_khr_gl_sharing",
            ctxp.GLX_DISPLAY_KHR: "cl_khr_gl_sharing",
            ctxp.WGL_HDC_KHR: "cl_khr_gl_sharing",
            ctxp.CGL_SHAREGROUP_KHR: "cl_khr_gl_sharing",
            }
        }

def doc_class(cls):
    print ".. class :: %s" % cls.__name__
    print
    if cls.__name__.startswith("gl_"):
        print "    Only available when PyOpenCL is compiled with GL support. See :func:`have_gl`."
        print

    cls_ext = ext_lookup.get(cls, {})
    for i in sorted(dir(cls)):
        if not i.startswith("_")  and not i == "to_string":
            print "    .. attribute :: %s" % i
            value = getattr(cls, i)

            if value in cls_ext:
                print
                print "        Available with the ``%s`` extension." % (
                        cls_ext[value])
                print

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
