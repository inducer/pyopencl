#include "pyhelper.h"

namespace py {
WrapFunc<int()> gc;
WrapFunc<void*(void*)> ref;
WrapFunc<void(void*)> deref;
WrapFunc<void(void*, cl_int)> call;
}

void
set_py_funcs(int (*_gc)(), void *(*_ref)(void*), void (*_deref)(void*),
             void (*_call)(void*, cl_int))
{
    py::gc = _gc;
    py::ref = _ref;
    py::deref = _deref;
    py::call = _call;
}
