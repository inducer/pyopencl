// Numpy import/init
//
// Copyright (C) 2009 Andreas Kloeckner
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.


#ifndef _FAYHVVAAA_PYOPENCL_HEADER_SEEN_NUMPY_INIT_HPP
#define _FAYHVVAAA_PYOPENCL_HEADER_SEEN_NUMPY_INIT_HPP


// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdexcept>


namespace
{
  static struct pyublas_array_importer
  {
    static bool do_import_array()
    {
#ifdef PYPY_VERSION
      import_array();
#else
      import_array1(false);
#endif
      return true;
    }

    pyublas_array_importer()
    {
      if (!do_import_array())
        throw std::runtime_error("numpy failed to initialize");
    }
  } _array_importer;
}




#endif
