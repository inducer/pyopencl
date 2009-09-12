import pyopencl as cl

def doc_class(cls):
    print ".. class :: %s" % cls.__name__
    print
    for i in sorted(dir(cls)):
        if not i.startswith("_")  and not i == "to_string":
            print "    .. attribute :: %s" % i
    print "    .. method :: to_string(value)"
    print
    print "        Returns a :class:`str` representing *value*."
    print
    print "        .. versionadded:: 0.91"
    print


print ".. This is an automatically generated file. DO NOT EDIT"
print
for cls in cl.CONSTANT_CLASSES:
    doc_class(cls)
