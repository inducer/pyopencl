#ifndef __PYOPENCL_ASYNC_H
#define __PYOPENCL_ASYNC_H

#include <functional>

namespace pyopencl {

// Start the helper thread
void init_async();
// Call @func in the helper thread
void call_async(const std::function<void()> &func);

}

#endif
