#include "async.h"
#include "function.h"

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace pyopencl {

template <typename T>
class Queue {
private:
    std::queue<T> m_queue;
    std::mutex m_mutex;
    std::condition_variable m_cond;
public:
    PYOPENCL_INLINE T
    pop()
    {
        std::unique_lock<std::mutex> mlock(m_mutex);
        while (m_queue.empty()) {
            m_cond.wait(mlock);
        }
        auto item = m_queue.front();
        m_queue.pop();
        return item;
    }
    PYOPENCL_INLINE void
    push(const T &item)
    {
        {
            // Sub scope for the lock
            std::unique_lock<std::mutex> mlock(m_mutex);
            m_queue.push(item);
        }
        m_cond.notify_one();
    }
};

class AsyncCaller {
private:
    Queue<std::function<void()> > m_queue;
    std::once_flag m_flag;
    void
    worker()
    {
        while (true) {
            try {
                auto func = m_queue.pop();
                func();
            } catch (...) {
            }
        }
    }
    void
    start_thread()
    {
        std::thread t(&AsyncCaller::worker, this);
        t.detach();
    }
public:
    PYOPENCL_INLINE void
    ensure_thread()
    {
        std::call_once(m_flag, &AsyncCaller::start_thread, this);
    }
    PYOPENCL_INLINE void
    push(const std::function<void()> &func)
    {
        ensure_thread();
        m_queue.push(func);
    }
};

static AsyncCaller async_caller;

void
call_async(const std::function<void()> &func)
{
    async_caller.push(func);
}

void
init_async()
{
    async_caller.ensure_thread();
}

}
