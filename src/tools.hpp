#ifndef _ASDFDAFVVAFF_PYCUDA_HEADER_SEEN_TOOLS_HPP
#define _ASDFDAFVVAFF_PYCUDA_HEADER_SEEN_TOOLS_HPP


#include <pybind11/pybind11.h>

#include <numeric>
#include "numpy_init.hpp"




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

    py::object gc_mod(
        py::handle<>(
          PyImport_ImportModule("gc")));
    gc_mod.attr("collect")();
  }


  // https://stackoverflow.com/a/28139075
  template <typename T>
  struct reversion_wrapper { T& iterable; };

  template <typename T>
  auto begin (reversion_wrapper<T> w) { return std::rbegin(w.iterable); }

  template <typename T>
  auto end (reversion_wrapper<T> w) { return std::rend(w.iterable); }

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
