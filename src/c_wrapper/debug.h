#include "wrap_cl.h"
#include "utils.h"

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

}

#endif
