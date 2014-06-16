#include "debug.h"
#include <stdlib.h>

namespace pyopencl {

static PYOPENCL_INLINE bool
_get_debug_env()
{
    const char *env = getenv("PYOPENCL_DEBUG");
#ifdef PYOPENCL_TRACE
    const bool default_debug = true;
#else
    const bool default_debug = false;
#endif
    if (!env) {
        return default_debug;
    }
    if (strcasecmp(env, "0") == 0 || strcasecmp(env, "f") == 0 ||
        strcasecmp(env, "false") == 0 || strcasecmp(env, "off") == 0) {
        return false;
    }
    if (strcasecmp(env, "1") == 0 || strcasecmp(env, "t") == 0 ||
        strcasecmp(env, "true") == 0 || strcasecmp(env, "on") == 0) {
        return true;
    }
    return default_debug;
}

bool debug_enabled = _get_debug_env();

}

int
get_debug()
{
    return (int)pyopencl::debug_enabled;
}

void
set_debug(int debug)
{
    pyopencl::debug_enabled = (bool)debug;
}
