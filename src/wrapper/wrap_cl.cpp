#include <cl.hpp>
#include "wrap_helpers.hpp"




namespace py = boost::python;




namespace
{
  py::handle<> 
    CLError, 
    CLMemoryError, 
    CLLogicError, 
    CLRuntimeError;




  void translate_cl_error(const cl::error &err)
  {
    if (err.code() == CL_MEM_OBJECT_ALLOCATION_FAILURE)
      PyErr_SetString(CLMemoryError.get(), err.what());
    else if (err.code() <= CL_INVALID_VALUE)
      PyErr_SetString(CLLogicError.get(), err.what());
    else if (err.code() > CL_INVALID_VALUE && err.code() < CL_SUCCESS)
      PyErr_SetString(CLRuntimeError.get(), err.what());
    else 
      PyErr_SetString(CLError.get(), err.what());
  }
}




BOOST_PYTHON_MODULE(_cl)
{
#define DECLARE_EXC(NAME, BASE) \
  CL##NAME = py::handle<>(PyErr_NewException("pyopencl._cl." #NAME, BASE, NULL)); \
  py::scope().attr(#NAME) = CL##NAME;

  {
    DECLARE_EXC(Error, NULL);
    py::tuple memerr_bases = py::make_tuple(
        CLError, 
        py::handle<>(py::borrowed(PyExc_MemoryError)));
    DECLARE_EXC(MemoryError, memerr_bases.ptr());
    DECLARE_EXC(LogicError, CLLogicError.get());
    DECLARE_EXC(RuntimeError, CLError.get());

    py::register_exception_translator<cl::error>(translate_cl_error);
  }

}
