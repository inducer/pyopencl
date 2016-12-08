# encoding: utf8
import numpy as __np

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

# {{{ cl -> np type mapping
cl2np_mapping = [
    (k, v) for k, v in globals().items() if not k.startswith('__')
    ]
# }}}