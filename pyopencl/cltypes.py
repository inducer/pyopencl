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

import numpy as np
from pyopencl.tools import get_or_register_dtype
import warnings

if __file__.endswith("array.py"):
    warnings.warn("pyopencl.array.vec is deprecated. Please use pyopencl.cltypes")

"""
This file provides a type mapping from OpenCl type names to their numpy equivalents
"""

char = np.int8
uchar = np.uint8
short = np.int16
ushort = np.uint16
int = np.int32
uint = np.uint32
long = np.int64
ulong = np.uint64
half = np.float16
float = np.float32
double = np.float64


# {{{ vector types

def _create_vector_types():
    _mapping = [(k, globals()[k]) for k in
                ["char", "uchar", "short", "ushort", "int",
                 "uint", "long", "ulong", "float", "double"]]

    def set_global(key, val):
        globals()[key] = val

    vec_types = {}
    vec_type_to_scalar_and_count = {}

    field_names = ["x", "y", "z", "w"]

    counts = [2, 3, 4, 8, 16]

    for base_name, base_type in _mapping:
        for count in counts:
            name = "%s%d" % (base_name, count)

            titles = field_names[:count]

            padded_count = count
            if count == 3:
                padded_count = 4

            names = ["s%d" % i for i in range(count)]
            while len(names) < padded_count:
                names.append("padding%d" % (len(names) - count))

            if len(titles) < len(names):
                titles.extend((len(names) - len(titles)) * [None])

            try:
                dtype = np.dtype(dict(
                    names=names,
                    formats=[base_type] * padded_count,
                    titles=titles))
            except NotImplementedError:
                try:
                    dtype = np.dtype([((n, title), base_type)
                                      for (n, title) in zip(names, titles)])
                except TypeError:
                    dtype = np.dtype([(n, base_type) for (n, title)
                                      in zip(names, titles)])

            get_or_register_dtype(name, dtype)

            set_global(name, dtype)

            def create_array(dtype, count, padded_count, *args, **kwargs):
                if len(args) < count:
                    from warnings import warn
                    warn("default values for make_xxx are deprecated;"
                         " instead specify all parameters or use"
                         " cltypes.zeros_xxx", DeprecationWarning)
                padded_args = tuple(list(args) + [0] * (padded_count - len(args)))
                array = eval("array(padded_args, dtype=dtype)",
                             dict(array=np.array, padded_args=padded_args,
                                  dtype=dtype))
                for key, val in list(kwargs.items()):
                    array[key] = val
                return array

            set_global("make_" + name, eval(
                "lambda *args, **kwargs: create_array(dtype, %i, %i, "
                "*args, **kwargs)" % (count, padded_count),
                dict(create_array=create_array, dtype=dtype)))
            set_global("filled_" + name, eval(
                "lambda val: make_%s(*[val]*%i)" % (name, count)))
            set_global("zeros_" + name, eval("lambda: filled_%s(0)" % (name)))
            set_global("ones_" + name, eval("lambda: filled_%s(1)" % (name)))

            vec_types[np.dtype(base_type), count] = dtype
            vec_type_to_scalar_and_count[dtype] = np.dtype(base_type), count

    return vec_types, vec_type_to_scalar_and_count


vec_types, vec_type_to_scalar_and_count = _create_vector_types()

# }}}

# vim: foldmethod=marker
