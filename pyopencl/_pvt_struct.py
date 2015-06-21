from __future__ import division
from __future__ import absolute_import

__copyright__ = """
Copyright (C) 2013 Marko Bencun
Copyright (C) 2014 Andreas Kloeckner
Copyright (C) 2014 Yichao Yu
"""

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

#import pyopencl._cl as _cl
import pyopencl.cffi_cl as _cl
from struct import pack as _pack

_size_t_char = ({
    8: 'Q',
    4: 'L',
    2: 'H',
    1: 'B',
})[_cl._ffi.sizeof('size_t')]
_type_char_map = {
    'n': _size_t_char.lower(),
    'N': _size_t_char
}
del _size_t_char
def pack(type_char, obj):
    if type_char == 'F':
        return _pack('f', obj.real) + _pack('f', obj.imag)
    elif type_char == "D":
        return _pack('d', obj.real) + _pack('d', obj.imag)
    else:
        return _pack(_type_char_map.get(type_char, type_char), obj)
