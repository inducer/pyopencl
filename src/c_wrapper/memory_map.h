#include "error.h"
#include "command_queue.h"
#include "memory_object.h"

#ifndef __PYOPENCL_MEMORY_MAP_H
#define __PYOPENCL_MEMORY_MAP_H

namespace pyopencl {

class event;

// {{{ memory_map

class memory_map : public clobj<void*> {
private:
    mutable volatile std::atomic_bool m_valid;
    command_queue m_queue;
    memory_object m_mem;
public:
    PYOPENCL_INLINE
    memory_map(const command_queue *queue, const memory_object *mem, void *ptr)
        : clobj(ptr), m_valid(true), m_queue(*queue), m_mem(*mem)
    {}
    ~memory_map();
    event *release(const command_queue *queue, const clobj_t *_wait_for,
                   uint32_t num_wait_for) const;
    generic_info get_info(cl_uint) const;
    intptr_t intptr() const;
    static memory_map *convert(clobj_t*, cl_event, command_queue *queue,
                               memory_object *buf, void *res);
};

// }}}

}

#endif
