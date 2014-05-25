#ifndef __PYOPENCL_ASYNC_H
#define __PYOPENCL_ASYNC_H

#include <functional>

namespace pyopencl {

void init_async();
void call_async(const std::function<void()> &func);

}

#endif
