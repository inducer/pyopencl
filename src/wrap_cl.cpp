// PyOpenCL-flavored C++ wrapper of the CL API
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


#define PY_ARRAY_UNIQUE_SYMBOL pyopencl_ARRAY_API

#include "wrap_cl.hpp"




using namespace pyopencl;




extern void pyopencl_expose_constants(py::module &m);
extern void pyopencl_expose_part_1(py::module &m);
extern void pyopencl_expose_part_2(py::module &m);
extern void pyopencl_expose_mempool(py::module &m);

static bool import_numpy_helper()
{
  import_array1(false);
  return true;
}

PYBIND11_MODULE(_cl, m)
{
  if (!import_numpy_helper())
    throw py::error_already_set();

  pyopencl_expose_constants(m);
  pyopencl_expose_part_1(m);
  pyopencl_expose_part_2(m);
  pyopencl_expose_mempool(m);
}

// vim: foldmethod=marker
