// Various odds and ends
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


#ifndef _ASDFDAFVVAFF_PYCUDA_HEADER_SEEN_TOOLS_HPP
#define _ASDFDAFVVAFF_PYCUDA_HEADER_SEEN_TOOLS_HPP


#include <pybind11/pybind11.h>

#include <numeric>
#include <numpy/arrayobject.h>



namespace pyopencl
{
  inline
  npy_intp size_from_dims(int ndim, const npy_intp *dims)
  {
    if (ndim != 0)
      return std::accumulate(dims, dims+ndim, 1, std::multiplies<npy_intp>());
    else
      return 1;
  }




  inline void run_python_gc()
  {
    namespace py = pybind11;

    py::module::import("gc").attr("collect")();
  }


  // https://stackoverflow.com/a/28139075
  template <typename T>
  struct reversion_wrapper { T& iterable; };

  template <typename T>
  auto begin (reversion_wrapper<T> w) { return w.iterable.rbegin(); }

  template <typename T>
  auto end (reversion_wrapper<T> w) { return w.iterable.rend(); }

  template <typename T>
  reversion_wrapper<T> reverse (T&& iterable) { return { iterable }; }


  // https://stackoverflow.com/a/44175911
  class noncopyable {
  public:
    noncopyable() = default;
    ~noncopyable() = default;

  private:
    noncopyable(const noncopyable&) = delete;
    noncopyable& operator=(const noncopyable&) = delete;
  };
}





#endif
