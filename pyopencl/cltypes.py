from __future__ import annotations


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

import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from pyopencl.tools import get_or_register_dtype


if TYPE_CHECKING:
    import builtins
    from collections.abc import MutableSequence

if __file__.endswith("array.py"):
    warnings.warn(
        "pyopencl.array.vec is deprecated. Please use pyopencl.cltypes.",
        stacklevel=2)

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

def _create_vector_types() -> tuple[
        dict[tuple[np.dtype[Any], builtins.int], np.dtype[Any]],
        dict[np.dtype[Any], tuple[np.dtype[Any], builtins.int]]]:
    mapping = [(k, globals()[k]) for k in
                ["char", "uchar", "short", "ushort", "int",
                 "uint", "long", "ulong", "float", "double"]]

    def set_global(key: str, val: np.dtype[Any]) -> None:
        globals()[key] = val

    vec_types: dict[tuple[np.dtype[Any], builtins.int], np.dtype[Any]] = {}
    vec_type_to_scalar_and_count: dict[np.dtype[Any],
                                       tuple[np.dtype[Any], builtins.int]] = {}

    field_names = ["x", "y", "z", "w"]

    counts = [2, 3, 4, 8, 16]

    for base_name, base_type in mapping:
        for count in counts:
            name = f"{base_name}{count}"
            titles = cast("MutableSequence[str | None]", field_names[:count])

            padded_count = count
            if count == 3:
                padded_count = 4

            names = [f"s{i}" for i in range(count)]
            while len(names) < padded_count:
                pad = len(names) - count
                names.append(f"padding{pad}")

            if len(titles) < len(names):
                pad = len(names) - len(titles)
                titles.extend([None] * pad)

            try:
                dtype = np.dtype({
                    "names": names,
                    "formats": [base_type] * padded_count,
                    "titles": titles})
            except NotImplementedError:
                try:
                    dtype = np.dtype([((n, title), base_type)
                                      for (n, title)
                                      in zip(names, titles, strict=True)])
                except TypeError:
                    dtype = np.dtype([(n, base_type) for n in names])

            assert isinstance(dtype, np.dtype)
            get_or_register_dtype(name, dtype)
            set_global(name, dtype)

            def create_array(dtype: np.dtype[Any],
                             count: int,
                             padded_count: int,
                             *args: Any, **kwargs: Any) -> dict[str, Any]:
                if len(args) < count:
                    from warnings import warn
                    warn("default values for make_xxx are deprecated;"
                         " instead specify all parameters or use"
                         " cltypes.zeros_xxx",
                         DeprecationWarning, stacklevel=4)

                padded_args = tuple(list(args) + [0] * (padded_count - len(args)))
                array = eval("array(padded_args, dtype=dtype)",
                             {"array": np.array,
                              "padded_args": padded_args,
                              "dtype": dtype})

                for key, val in kwargs.items():
                    array[key] = val

                return array

            set_global(
                f"make_{name}",
                eval("lambda *args, **kwargs: "
                     f"create_array(dtype, {count}, {padded_count}, *args, **kwargs)",
                     {"create_array": create_array, "dtype": dtype}))
            set_global(
                f"filled_{name}",
                eval(f"lambda val: make_{name}(*[val]*{count})"))
            set_global(f"zeros_{name}", eval(f"lambda: filled_{name}(0)"))
            set_global(f"ones_{name}", eval(f"lambda: filled_{name}(1)"))

            base_dtype = np.dtype(base_type)
            vec_types[base_dtype, count] = dtype
            vec_type_to_scalar_and_count[dtype] = base_dtype, count

    return vec_types, vec_type_to_scalar_and_count


vec_types, vec_type_to_scalar_and_count = _create_vector_types()

# }}}

char2: np.dtype[Any]
char3: np.dtype[Any]
char4: np.dtype[Any]
char8: np.dtype[Any]
char16: np.dtype[Any]

uchar2: np.dtype[Any]
uchar3: np.dtype[Any]
uchar4: np.dtype[Any]
uchar8: np.dtype[Any]
uchar16: np.dtype[Any]

short2: np.dtype[Any]
short3: np.dtype[Any]
short4: np.dtype[Any]
short8: np.dtype[Any]
short16: np.dtype[Any]

ushort2: np.dtype[Any]
ushort3: np.dtype[Any]
ushort4: np.dtype[Any]
ushort8: np.dtype[Any]
ushort16: np.dtype[Any]

int2: np.dtype[Any]
int3: np.dtype[Any]
int4: np.dtype[Any]
int8: np.dtype[Any]
int16: np.dtype[Any]

uint2: np.dtype[Any]
uint3: np.dtype[Any]
uint4: np.dtype[Any]
uint8: np.dtype[Any]
uint16: np.dtype[Any]

long2: np.dtype[Any]
long3: np.dtype[Any]
long4: np.dtype[Any]
long8: np.dtype[Any]
long16: np.dtype[Any]

ulong2: np.dtype[Any]
ulong3: np.dtype[Any]
ulong4: np.dtype[Any]
ulong8: np.dtype[Any]
ulong16: np.dtype[Any]

float2: np.dtype[Any]
float3: np.dtype[Any]
float4: np.dtype[Any]
float8: np.dtype[Any]
float16: np.dtype[Any]

double2: np.dtype[Any]
double3: np.dtype[Any]
double4: np.dtype[Any]
double8: np.dtype[Any]
double16: np.dtype[Any]

# vim: foldmethod=marker
