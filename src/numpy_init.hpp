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
      import_array1(false);
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
