#include "wrap_cl.hpp"




using namespace pyopencl;




extern void pyopencl_expose_constants();
extern void pyopencl_expose_part_1();
extern void pyopencl_expose_part_2();
extern void pyopencl_expose_mempool();

BOOST_PYTHON_MODULE(_cl)
{
  pyopencl_expose_constants();
  pyopencl_expose_part_1();
  pyopencl_expose_part_2();
  pyopencl_expose_mempool();
}

// vim: foldmethod=marker
