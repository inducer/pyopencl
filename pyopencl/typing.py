from __future__ import annotations


__copyright__ = "Copyright (C) 2025 University of Illinois Board of Trustees"

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

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias, TypeVar

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Buffer as abc_Buffer


if TYPE_CHECKING:
    import pyopencl as _cl
    import pyopencl.array as _cl_array


DTypeT = TypeVar("DTypeT", bound=np.dtype[Any])

HasBufferInterface: TypeAlias = abc_Buffer | NDArray[Any]
SVMInnerT = TypeVar("SVMInnerT", bound=HasBufferInterface)
WaitList: TypeAlias = Sequence["_cl.Event"] | None
KernelArg: TypeAlias = """
    int
    | float
    | complex
    | HasBufferInterface
    | np.generic
    | _cl.Buffer
    | _cl.Image
    | _cl.Sampler
    | _cl.SVMPointer
    | _cl_array.Array
    | None"""

Allocator: TypeAlias = "Callable[[int], _cl.MemoryObjectHolder | _cl.SVMPointer]"
