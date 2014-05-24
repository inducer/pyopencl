#include "utils.h"
#include "error.h"
#include <atomic>

namespace pyopencl {

static std::atomic<unsigned long> pyobj_id = ATOMIC_VAR_INIT(1ul);
unsigned long
next_obj_id()
{
    unsigned long id;
    do {
        id = std::atomic_fetch_add(&pyobj_id, 1ul);
    } while (id == 0);
    return id;
}

static int
dummy_python_gc()
{
    return 0;
}

static void
dummy_python_deref(unsigned long)
{
}

int (*python_gc)() = dummy_python_gc;
void (*python_deref)(unsigned long) = dummy_python_deref;

}

void
set_gc(int (*func)())
{
    pyopencl::python_gc = func ? func : pyopencl::dummy_python_gc;
}

void
set_deref(void (*func)(unsigned long))
{
    pyopencl::python_deref = func ? func : pyopencl::dummy_python_deref;
}
