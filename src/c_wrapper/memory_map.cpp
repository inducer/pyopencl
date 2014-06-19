#include "memory_map.h"
#include "image.h"
#include "buffer.h"
#include "event.h"
#include "clhelper.h"

namespace pyopencl {

template class clobj<void*>;
template void print_arg<void*>(std::ostream&, void *const&, bool);
template void print_buf<void*>(std::ostream&, void *const*,
                               size_t, ArgType, bool, bool);

memory_map::~memory_map()
{
    if (!m_valid.exchange(false))
        return;
    pyopencl_call_guarded_cleanup(clEnqueueUnmapMemObject, m_queue,
                                  m_mem, this, 0, nullptr, nullptr);
}

void
memory_map::release(clobj_t *evt, const command_queue *queue,
                    const clobj_t *_wait_for, uint32_t num_wait_for) const
{
    if (!m_valid.exchange(false)) {
        throw clerror("MemoryMap.release", CL_INVALID_VALUE,
                      "trying to double-unref mem map");
    }
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    queue = queue ? queue : &m_queue;
    pyopencl_call_guarded(clEnqueueUnmapMemObject, queue,
                          m_mem, this, wait_for, event_out(evt));
}

generic_info
memory_map::get_info(cl_uint) const
{
    throw clerror("MemoryMap.get_info", CL_INVALID_VALUE);
}

intptr_t
memory_map::intptr() const
{
    return m_valid ? (intptr_t)data() : 0;
}

memory_map*
convert_memory_map(clobj_t evt, command_queue *queue,
                   memory_object *buf, void *res)
{
    try {
        return new memory_map(queue, buf, res);
    } catch (...) {
        delete evt;
        pyopencl_call_guarded_cleanup(clEnqueueUnmapMemObject, queue,
                                      buf, res, 0, nullptr, nullptr);
        throw;
    }
}

}

// c wrapper
// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

// Memory Map
error*
memory_map__release(clobj_t _map, clobj_t _queue, const clobj_t *_wait_for,
                    uint32_t num_wait_for, clobj_t *evt)
{
    auto map = static_cast<memory_map*>(_map);
    auto queue = static_cast<command_queue*>(_queue);
    return c_handle_error([&] {
            map->release(evt, queue, _wait_for, num_wait_for);
        });
}

void*
memory_map__data(clobj_t _map)
{
    return static_cast<memory_map*>(_map)->data();
}

error*
enqueue_map_image(clobj_t *evt, clobj_t *map, clobj_t _queue, clobj_t _mem,
                  cl_map_flags flags, const size_t *_origin, size_t origin_l,
                  const size_t *_region, size_t region_l, size_t *row_pitch,
                  size_t *slice_pitch, const clobj_t *_wait_for,
                  uint32_t num_wait_for, int block)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto img = static_cast<image*>(_mem);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    ConstBuffer<size_t, 3> origin(_origin, origin_l);
    ConstBuffer<size_t, 3> region(_region, region_l);
    return c_handle_error([&] {
            void *res = retry_mem_error([&] {
                    return pyopencl_call_guarded(
                        clEnqueueMapImage, queue, img, bool(block), flags,
                        origin, region, row_pitch, slice_pitch, wait_for,
                        event_out(evt));
                });
            *map = convert_memory_map(*evt, queue, img, res);
        });
}

error*
enqueue_map_buffer(clobj_t *evt, clobj_t *map, clobj_t _queue, clobj_t _mem,
                   cl_map_flags flags, size_t offset, size_t size,
                   const clobj_t *_wait_for, uint32_t num_wait_for,
                   int block)
{
    auto queue = static_cast<command_queue*>(_queue);
    auto buf = static_cast<buffer*>(_mem);
    const auto wait_for = buf_from_class<event>(_wait_for, num_wait_for);
    return c_handle_error([&] {
            void *res = retry_mem_error([&] {
                    return pyopencl_call_guarded(
                        clEnqueueMapBuffer, queue, buf, bool(block),
                        flags, offset, size, wait_for, event_out(evt));
                });
            *map = convert_memory_map(*evt, queue, buf, res);
        });
}
