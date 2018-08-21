#include "wrap_cl.hpp"




using namespace pyopencl;




extern void pyopencl_expose_constants(py::module &m);
extern void pyopencl_expose_part_1(py::module &m);
extern void pyopencl_expose_part_2(py::module &m);
extern void pyopencl_expose_mempool(py::module &m);

PYBIND11_MODULE(_cl, m)
{
  pyopencl_expose_constants(m);
  pyopencl_expose_part_1(m);
  pyopencl_expose_part_2(m);
  pyopencl_expose_mempool(m);
}

// vim: foldmethod=marker
