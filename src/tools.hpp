#ifndef _ASDFDAFVVAFF_PYCUDA_HEADER_SEEN_TOOLS_HPP
#define _ASDFDAFVVAFF_PYCUDA_HEADER_SEEN_TOOLS_HPP




#include <boost/python.hpp>
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
    namespace py = boost::python;

    py::object gc_mod(
        py::handle<>(
          PyImport_ImportModule("gc")));
    gc_mod.attr("collect")();
  }
}





#endif
