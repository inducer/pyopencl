#include "utils.h"
#include "error.h"
#include <atomic>

namespace pyopencl {

static int
dummy_python_gc()
{
    return 0;
}

static void
dummy_python_ref_func(void*)
{
}

int (*python_gc)() = dummy_python_gc;
void (*python_deref)(void*) = dummy_python_ref_func;
void (*python_ref)(void*) = dummy_python_ref_func;

}

void
set_gc(int (*func)())
{
    pyopencl::python_gc = func ? func : pyopencl::dummy_python_gc;
}

void
set_ref_funcs(void (*ref)(void*), void (*deref)(void*))
{
    pyopencl::python_ref = ref ? ref : pyopencl::dummy_python_ref_func;
    pyopencl::python_deref = deref ? deref : pyopencl::dummy_python_ref_func;
}
