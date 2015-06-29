#include "debug.h"
#include <iostream>
#include <ios>
#include <iomanip>
#include <stdlib.h>

std::mutex dbg_lock;

void
dbg_print_str(std::ostream &stm, const char *str, size_t len)
{
    stm << '"';
    for (size_t i = 0;i < len;i++) {
        char escaped = 0;
#define escape_char(in, out)                    \
        case in:                                \
            escaped = out;                      \
            break
        switch (str[i]) {
            escape_char('\'', '\'');
            escape_char('\"', '\"');
            escape_char('\?', '\?');
            escape_char('\\', '\\');
            escape_char('\0', '0');
            escape_char('\a', 'a');
            escape_char('\b', 'b');
            escape_char('\f', 'f');
            escape_char('\r', 'r');
            escape_char('\v', 'v');
        default:
            break;
        }
        if (escaped) {
            stm << '\\' << escaped;
        } else {
            stm << str[i];
        }
    }
    stm << '"';
}

void
dbg_print_bytes(std::ostream &stm, const unsigned char *bytes, size_t len)
{
    stm << '"';
    for (size_t i = 0;i < len;i++) {
        stm << "\\x" << std::hex << std::setfill('0')
            << std::setw(2) << bytes[i];
    }
    stm << std::dec << '"';
}

static PYOPENCL_INLINE bool
_get_debug_env()
{
    const char *env = getenv("PYOPENCL_DEBUG");
    const bool default_debug = DEFAULT_DEBUG;
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

int
get_debug()
{
    return (int) debug_enabled;
}

void
set_debug(int debug)
{
    debug_enabled = (bool)debug;
}
