import pyopencl as cl

def doc_class(cls):
    print ".. class :: %s" % cls.__name__
    print
    for i in sorted(dir(cls)):
        if not i.startswith("_"):
            print "    .. attribute :: %s" % i
    print


print ".. This is an automatically generated file. DO NOT EDIT"
for cls in cl.CONSTANT_CLASSES:
    doc_class(cls)
