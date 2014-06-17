#include "error.h"
#include <atomic>

#ifndef __PYOPENCL_MEMORY_OBJECT_H
#define __PYOPENCL_MEMORY_OBJECT_H

namespace pyopencl {

// {{{ memory_object

extern template class clobj<cl_mem>;

class memory_object : public clobj<cl_mem> {
private:
    mutable volatile std::atomic_bool m_valid;
public:
    constexpr static const char *class_name = "MEMORY_OBJECT";
    PYOPENCL_INLINE
    memory_object(cl_mem mem, bool retain)
        : clobj(mem), m_valid(true)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainMemObject, this);
        }
    }
    PYOPENCL_INLINE
    memory_object(const memory_object &mem)
        : memory_object(mem.data(), true)
    {}
    ~memory_object();
    generic_info get_info(cl_uint param_name) const;
    void
    release() const
    {
        if (PYOPENCL_UNLIKELY(!m_valid.exchange(false))) {
            throw clerror("MemoryObject.release", CL_INVALID_VALUE,
                          "trying to double-unref mem object");
        }
        pyopencl_call_guarded(clReleaseMemObject, this);
    }
#if 0
    PYOPENCL_USE_RESULT size_t
    size() const
    {
        size_t param_value;
        pyopencl_call_guarded(clGetMemObjectInfo, this, CL_MEM_SIZE,
                              make_sizearg(param_value), nullptr);
        return param_value;
    }
#endif
};

// #if PYOPENCL_CL_VERSION >= 0x1020
//   PYOPENCL_INLINE
//   event *enqueue_migrate_mem_objects(
//       command_queue &cq,
//       py::object py_mem_objects,
//       cl_mem_migration_flags flags,
//       py::object py_wait_for)
//   {
//     PYOPENCL_PARSE_WAIT_FOR;

//     std::vector<cl_mem> mem_objects;
//     PYTHON_FOREACH(mo, py_mem_objects)
//       mem_objects.push_back(py::extract<memory_object &>(mo)().data());

//     cl_event evt;
//     PYOPENCL_RETRY_IF_MEM_ERROR(
//       PYOPENCL_CALL_GUARDED(clEnqueueMigrateMemObjects, (
//             cq.data(),
//             mem_objects.size(), mem_objects.empty( ) ? nullptr : &mem_objects.front(),
//             flags,
//             PYOPENCL_WAITLIST_ARGS, &evt
//             ));
//       );
//     PYOPENCL_RETURN_NEW_EVENT(evt);
//   }
// #endif

// #ifdef cl_ext_migrate_memobject
//   PYOPENCL_INLINE
//   event *enqueue_migrate_mem_object_ext(
//       command_queue &cq,
//       py::object py_mem_objects,
//       cl_mem_migration_flags_ext flags,
//       py::object py_wait_for)
//   {
//     PYOPENCL_PARSE_WAIT_FOR;

// #if PYOPENCL_CL_VERSION >= 0x1020
//     // {{{ get platform
//     cl_device_id dev;
//     PYOPENCL_CALL_GUARDED(clGetCommandQueueInfo, (cq.data(), CL_QUEUE_DEVICE,
//           sizeof(dev), &dev, nullptr));
//     cl_platform_id plat;
//     PYOPENCL_CALL_GUARDED(clGetDeviceInfo, (cq.data(), CL_DEVICE_PLATFORM,
//           sizeof(plat), &plat, nullptr));
//     // }}}
// #endif

//     PYOPENCL_GET_EXT_FUN(plat,
//         clEnqueueMigrateMemObjectEXT, enqueue_migrate_fn);

//     std::vector<cl_mem> mem_objects;
//     PYTHON_FOREACH(mo, py_mem_objects)
//       mem_objects.push_back(py::extract<memory_object &>(mo)().data());

//     cl_event evt;
//     PYOPENCL_RETRY_IF_MEM_ERROR(
//       PYOPENCL_CALL_GUARDED(enqueue_migrate_fn, (
//             cq.data(),
//             mem_objects.size(), mem_objects.empty( ) ? nullptr : &mem_objects.front(),
//             flags,
//             PYOPENCL_WAITLIST_ARGS, &evt
//             ));
//       );
//     PYOPENCL_RETURN_NEW_EVENT(evt);
//   }
// #endif

// }}}

}

#endif
