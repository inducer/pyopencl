#include "wrap_cl.h"
#include "function.h"
#include <string.h>
#include <mutex>

#ifndef __PYOPENCL_DEBUG_H
#define __PYOPENCL_DEBUG_H

namespace pyopencl {

extern bool debug_enabled;
#ifdef PYOPENCL_TRACE
#define DEFAULT_DEBUG true
#else
#define DEFAULT_DEBUG false
#endif

#define DEBUG_ON (PYOPENCL_EXPECT(debug_enabled, DEFAULT_DEBUG))

extern std::mutex dbg_lock;

void dbg_print_str(std::ostream&, const char*, size_t);
static PYOPENCL_INLINE void
dbg_print_str(std::ostream &stm, const char *str)
{
    return dbg_print_str(stm, str, strlen(str));
}

}

#endif
