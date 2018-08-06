#ifndef PYCUDA_WRAP_HELPERS_HEADER_SEEN
#define PYCUDA_WRAP_HELPERS_HEADER_SEEN




#include <boost/version.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>




namespace py = boost::python;




#if (BOOST_VERSION/100) < 1035
#warning *******************************************************************
#warning **** Your version of Boost C++ is likely too old for PyOpenCL. ****
#warning *******************************************************************
#endif




#define PYTHON_ERROR(TYPE, REASON) \
{ \
  PyErr_SetString(PyExc_##TYPE, REASON); \
  throw boost::python::error_already_set(); \
}

#define ENUM_VALUE(NAME) \
  value(#NAME, NAME)

#define DEF_SIMPLE_METHOD(NAME) \
  def(#NAME, &cls::NAME)

#define DEF_SIMPLE_METHOD_WITH_ARGS(NAME, ARGS) \
  def(#NAME, &cls::NAME, boost::python::args ARGS)

#define DEF_SIMPLE_FUNCTION(NAME) \
  boost::python::def(#NAME, &NAME)

#define DEF_SIMPLE_FUNCTION_WITH_ARGS(NAME, ARGS) \
  boost::python::def(#NAME, &NAME, boost::python::args ARGS)

#define DEF_SIMPLE_RO_MEMBER(NAME) \
  def_readonly(#NAME, &cls::m_##NAME)

#define DEF_SIMPLE_RW_MEMBER(NAME) \
  def_readwrite(#NAME, &cls::m_##NAME)

#define PYTHON_FOREACH(NAME, ITERABLE) \
  BOOST_FOREACH(boost::python::object NAME, \
      std::make_pair( \
        boost::python::stl_input_iterator<boost::python::object>(ITERABLE), \
        boost::python::stl_input_iterator<boost::python::object>()))

#define COPY_PY_LIST(TYPE, NAME) \
  std::copy( \
      boost::python::stl_input_iterator<TYPE>(py_##NAME), \
      boost::python::stl_input_iterator<TYPE>(), \
      std::back_inserter(NAME));

#define COPY_PY_COORD_TRIPLE(NAME) \
  size_t NAME[3] = {0, 0, 0}; \
  { \
    size_t my_len = len(py_##NAME); \
    if (my_len > 3) \
      throw error("transfer", CL_INVALID_VALUE, #NAME "has too many components"); \
    for (size_t i = 0; i < my_len; ++i) \
      NAME[i] = py::extract<size_t>(py_##NAME[i])(); \
  }

#define COPY_PY_PITCH_TUPLE(NAME) \
  size_t NAME[2] = {0, 0}; \
  if (py_##NAME.ptr() != Py_None) \
  { \
    size_t my_len = len(py_##NAME); \
    if (my_len > 2) \
      throw error("transfer", CL_INVALID_VALUE, #NAME "has too many components"); \
    for (size_t i = 0; i < my_len; ++i) \
      NAME[i] = py::extract<size_t>(py_##NAME[i])(); \
  }

#define COPY_PY_REGION_TRIPLE(NAME) \
  size_t NAME[3] = {1, 1, 1}; \
  { \
    size_t my_len = len(py_##NAME); \
    if (my_len > 3) \
      throw error("transfer", CL_INVALID_VALUE, #NAME "has too many components"); \
    for (size_t i = 0; i < my_len; ++i) \
      NAME[i] = py::extract<size_t>(py_##NAME[i])(); \
  }

#define PYOPENCL_PARSE_NUMPY_ARRAY_SPEC \
    PyArray_Descr *tp_descr; \
    if (PyArray_DescrConverter(dtype.ptr(), &tp_descr) != NPY_SUCCEED) \
      throw py::error_already_set(); \
    \
    py::extract<npy_intp> shape_as_int(py_shape); \
    std::vector<npy_intp> shape; \
    \
    if (shape_as_int.check()) \
      shape.push_back(shape_as_int()); \
    else \
      COPY_PY_LIST(npy_intp, shape); \
    \
    NPY_ORDER order = PyArray_CORDER; \
    PyArray_OrderConverter(py_order.ptr(), &order); \
    \
    int ary_flags = 0; \
    if (order == PyArray_FORTRANORDER) \
      ary_flags |= NPY_FARRAY; \
    else if (order == PyArray_CORDER) \
      ary_flags |= NPY_CARRAY; \
    else \
      throw std::runtime_error("unrecognized order specifier"); \
    \
    std::vector<npy_intp> strides; \
    if (py_strides.ptr() != Py_None) \
    { \
      COPY_PY_LIST(npy_intp, strides); \
    }

#define PYOPENCL_RETURN_VECTOR(ITEMTYPE, NAME) \
  { \
    py::list pyopencl_result; \
    BOOST_FOREACH(ITEMTYPE item, NAME) \
      pyopencl_result.append(item); \
    return pyopencl_result; \
  }

namespace
{
  template <typename T>
  inline boost::python::handle<> handle_from_new_ptr(T *ptr)
  {
    return boost::python::handle<>(
        typename boost::python::manage_new_object::apply<T *>::type()(ptr));
  }

  template <typename T, typename ClType>
  inline T *from_int_ptr(intptr_t obj_ref)
  {
    ClType clobj = (ClType) obj_ref;
    return new T(clobj, /* retain */ true);
  }

  template <typename T>
  inline intptr_t to_int_ptr(T const &obj)
  {
    return (intptr_t) obj.data();
  }
}

#define PYOPENCL_EXPOSE_TO_FROM_INT_PTR(CL_TYPENAME) \
  .def("from_int_ptr", from_int_ptr<cls, CL_TYPENAME>, \
      py::return_value_policy<py::manage_new_object>(), \
      py::arg("int_ptr_value"), \
      "(static method) Return a new Python object referencing the C-level " \
      ":c:type:`" #CL_TYPENAME "` object at the location pointed to " \
      "by *int_ptr_value*. The relevant :c:func:`clRetain*` function " \
      "will be called." \
      "\n\n.. versionadded:: 2013.2\n") \
  .staticmethod("from_int_ptr") \
  .add_property("int_ptr", to_int_ptr<cls>, \
      "Return an integer corresponding to the pointer value " \
      "of the underlying :c:type:`" #CL_TYPENAME "`. " \
      "Use :meth:`from_int_ptr` to turn back into a Python object." \
      "\n\n.. versionadded:: 2013.2\n") \

#endif
