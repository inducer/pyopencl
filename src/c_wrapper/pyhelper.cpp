#include "pyhelper.h"

namespace pyopencl {

namespace py {
WrapFunc<int()> gc;
WrapFunc<void(void*)> ref;
WrapFunc<void(void*)> deref;
WrapFunc<void(void*, cl_int)> call;
}

}

void
set_py_funcs(int (*_gc)(), void (*_ref)(void*), void (*_deref)(void*),
             void (*_call)(void*, cl_int))
{
    pyopencl::py::gc = _gc;
    pyopencl::py::ref = _ref;
    pyopencl::py::deref = _deref;
    pyopencl::py::call = _call;
}
