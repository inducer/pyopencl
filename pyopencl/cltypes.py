# encoding: utf8

__copyright__ = "Copyright (C) 2016 Jonathan Mackenzie"

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

import numpy as __np
from pyopencl.tools import get_or_register_dtype
import warnings

if __file__.endswith('array.py'):
    warnings.warn("pyopencl.array.vec is deprecated. Please use pyopencl.cltypes")

"""
This file provides a type mapping from OpenCl type names to their numpy equivalents
"""

char = __np.int8
uchar = __np.uint8
short = __np.int16
ushort = __np.uint16
int = __np.int32
uint = __np.uint32
long = __np.int64
ulong = __np.uint64
float = __np.float32
double = __np.float64

# {{{ vector types


def _create_vector_types():
    def set_global(key, val):
        globals()[key] = val
    field_names = ["x", "y", "z", "w"]

    set_global('types', {})
    set_global('type_to_scalar_and_count', {})

    counts = [2, 3, 4, 8, 16]

    for base_name, base_type in [(k, v) for k, v in globals().items() if not k.startswith('__')]:
        for count in counts:
            name = "%s%d" % (base_name, count)

            titles = field_names[:count]

            padded_count = count
            if count == 3:
                padded_count = 4

            names = ["s%d" % i for i in range(count)]
            while len(names) < padded_count:
                names.append("padding%d" % (len(names)-count))

            if len(titles) < len(names):
                titles.extend((len(names)-len(titles))*[None])

            try:
                dtype = __np.dtype(dict(
                    names=names,
                    formats=[base_type]*padded_count,
                    titles=titles))
            except NotImplementedError:
                try:
                    dtype = __np.dtype([((n, title), base_type)
                                      for (n, title) in zip(names, titles)])
                except TypeError:
                    dtype = __np.dtype([(n, base_type) for (n, title)
                                      in zip(names, titles)])

            get_or_register_dtype(name, dtype)

            set_global(name, dtype)

            def create_array(dtype, count, padded_count, *args, **kwargs):
                if len(args) < count:
                    from warnings import warn
                    warn("default values for make_xxx are deprecated;"
                            " instead specify all parameters or use"
                            " array.vec.zeros_xxx", DeprecationWarning)
                padded_args = tuple(list(args)+[0]*(padded_count-len(args)))
                array = eval("array(padded_args, dtype=dtype)",
                        dict(array=__np.array, padded_args=padded_args,
                        dtype=dtype))
                for key, val in list(kwargs.items()):
                    array[key] = val
                return array

            set_global("make_"+name, staticmethod(eval(
                    "lambda *args, **kwargs: create_array(dtype, %i, %i, "
                    "*args, **kwargs)" % (count, padded_count),
                    dict(create_array=create_array, dtype=dtype))))
            set_global("filled_"+name, staticmethod(eval(
                    "lambda val: vec.make_%s(*[val]*%i)" % (name, count))))
            set_global("zeros_"+name,
                    staticmethod(eval("lambda: vec.filled_%s(0)" % (name))))
            set_global("ones_"+name,
                    staticmethod(eval("lambda: vec.filled_%s(1)" % (name))))

            globals()['types'][__np.dtype(base_type), count] = dtype
            globals()['type_to_scalar_and_count'][dtype] = __np.dtype(base_type), count


# }}}


half = __np.float16

