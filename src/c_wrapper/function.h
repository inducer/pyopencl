#include <functional>

#ifndef __PYOPENCL_FUNCTION_H
#define __PYOPENCL_FUNCTION_H

#if defined __GNUC__ &&  __GNUC__ > 3
#define PYOPENCL_INLINE inline __attribute__((__always_inline__))
#else
#define PYOPENCL_INLINE inline
#endif

#endif
