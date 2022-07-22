// Wrap memory pool
//
// Copyright (C) 2009 Andreas Kloeckner
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.


// Gregor Thalhammer (on Apr 13, 2011) said it's necessary to import Python.h
// first to prevent OS X from overriding a bunch of macros. (e.g. isspace)
#include <Python.h>

#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL pyopencl_ARRAY_API

#include <memory>
#include <vector>
#include "wrap_helpers.hpp"
#include "wrap_cl.hpp"
#include "mempool.hpp"
#include "tools.hpp"



namespace
{
  class test_allocator
  {
    public:
      typedef void *pointer_type;
      typedef size_t size_type;

      virtual bool is_deferred() const
      {
        return false;
      }

      virtual pointer_type allocate(size_type s)
      {
        return nullptr;
      }

      virtual pointer_type hand_out_existing_block(pointer_type &&p)
      {
        return p;
      }

      void free(pointer_type &&p)
      { }

      void try_release_blocks()
      { }
  };


  // {{{ buffer allocators

  class buffer_allocator_base
  {
    protected:
      std::shared_ptr<pyopencl::context> m_context;
      cl_mem_flags m_flags;

    public:
      buffer_allocator_base(std::shared_ptr<pyopencl::context> const &ctx,
          cl_mem_flags flags=CL_MEM_READ_WRITE)
        : m_context(ctx), m_flags(flags)
      {
        if (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))
          throw pyopencl::error("Allocator", CL_INVALID_VALUE,
              "cannot specify USE_HOST_PTR or COPY_HOST_PTR flags");
      }

      buffer_allocator_base(buffer_allocator_base const &src)
      : m_context(src.m_context), m_flags(src.m_flags)
      { }

      virtual ~buffer_allocator_base()
      { }

      typedef cl_mem pointer_type;
      typedef size_t size_type;

      virtual bool is_deferred() const = 0;
      virtual pointer_type allocate(size_type s) = 0;

      virtual pointer_type hand_out_existing_block(pointer_type p)
      {
        return p;
      }

      void free(pointer_type &&p)
      {
        PYOPENCL_CALL_GUARDED(clReleaseMemObject, (p));
      }

      void try_release_blocks()
      {
        pyopencl::run_python_gc();
      }
  };


  class deferred_buffer_allocator : public buffer_allocator_base
  {
    private:
      typedef buffer_allocator_base super;

    public:
      deferred_buffer_allocator(std::shared_ptr<pyopencl::context> const &ctx,
          cl_mem_flags flags=CL_MEM_READ_WRITE)
        : super(ctx, flags)
      { }

      bool is_deferred() const
      { return true; }

      pointer_type allocate(size_type s)
      {
        if (s == 0)
          return nullptr;

        return pyopencl::create_buffer(m_context->data(), m_flags, s, 0);
      }
  };

  const unsigned zero = 0;

  class immediate_buffer_allocator : public buffer_allocator_base
  {
    private:
      typedef buffer_allocator_base super;
      pyopencl::command_queue m_queue;

    public:
      immediate_buffer_allocator(pyopencl::command_queue &queue,
          cl_mem_flags flags=CL_MEM_READ_WRITE)
        : super(std::shared_ptr<pyopencl::context>(queue.get_context()), flags),
        m_queue(queue.data(), /*retain*/ true)
      { }

      immediate_buffer_allocator(immediate_buffer_allocator const &src)
        : super(src), m_queue(src.m_queue)
      { }

      bool is_deferred() const
      { return false; }

      pointer_type allocate(size_type s)
      {
        if (s == 0)
          return nullptr;

        pointer_type ptr =  pyopencl::create_buffer(
            m_context->data(), m_flags, s, 0);

        // Make sure the buffer gets allocated right here and right now.
        // This looks (and is) expensive. But immediate allocators
        // have their main use in memory pools, whose basic assumption
        // is that allocation is too expensive anyway--but they rely
        // on 'out-of-memory' being reported on allocation. (If it is
        // reported in a deferred manner, it has no way to react
        // (e.g. by freeing unused memory) because it is not part of
        // the call stack.)
        if (m_queue.get_hex_device_version() < 0x1020)
        {
          unsigned zero = 0;
          PYOPENCL_CALL_GUARDED(clEnqueueWriteBuffer, (
                m_queue.data(),
                ptr,
                /* is blocking */ CL_FALSE,
                0, std::min(s, sizeof(zero)), &zero,
                0, NULL, NULL
                ));
        }
        else
        {
          PYOPENCL_CALL_GUARDED(clEnqueueMigrateMemObjects, (
                m_queue.data(),
                1, &ptr, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                0, NULL, NULL
                ));
        }

        // No need to wait for completion here. clWaitForEvents (e.g.)
        // cannot return mem object allocation failures. This implies that
        // the buffer is faulted onto the device on enqueue.

        return ptr;
      }
  };

  // }}}


  // {{{ buffer_allocator_call

  inline
  pyopencl::buffer *buffer_allocator_call(buffer_allocator_base &alloc, size_t size)
  {
    cl_mem mem;
    int try_count = 0;
    while (try_count < 2)
    {
      try
      {
        mem = alloc.allocate(size);
        break;
      }
      catch (pyopencl::error &e)
      {
        if (!e.is_out_of_memory())
          throw;
        if (++try_count == 2)
          throw;
      }

      alloc.try_release_blocks();
    }

    if (!mem)
    {
      if (size == 0)
        return nullptr;
      else
        throw pyopencl::error("Allocator", CL_INVALID_VALUE,
            "allocator succeeded but returned NULL cl_mem");
    }

    try
    {
      return new pyopencl::buffer(mem, false);
    }
    catch (...)
    {
      PYOPENCL_CALL_GUARDED(clReleaseMemObject, (mem));
      throw;
    }
  }

  // }}}


  // {{{ pooled_buffer

  class pooled_buffer
    : public pyopencl::pooled_allocation<pyopencl::memory_pool<buffer_allocator_base> >,
    public pyopencl::memory_object_holder
  {
    private:
      typedef
        pyopencl::pooled_allocation<pyopencl::memory_pool<buffer_allocator_base> >
        super;

    public:
      pooled_buffer(
          std::shared_ptr<super::pool_type> p, super::size_type s)
        : super(p, s)
      { }

      const super::pointer_type data() const
      { return m_ptr; }

      size_t size() const
      {
        return m_size;
      }
  };

  // }}}


  // {{{ buffer_pool_allocate

  pooled_buffer *buffer_pool_allocate(
      std::shared_ptr<pyopencl::memory_pool<buffer_allocator_base> > pool,
      pyopencl::memory_pool<buffer_allocator_base>::size_type sz)
  {
    return new pooled_buffer(pool, sz);
  }

  // }}}


#if PYOPENCL_CL_VERSION >= 0x2000

  // FIXME: Does this need deferred and immediate just like the buffer-level
  // allocators? (I.e. can I tell whether I am out of memory just from allocations?)

  struct svm_held_pointer
  {
    void *ptr;
    pyopencl::command_queue_ref queue;
  };


  // {{{ svm allocator

  class svm_allocator
  {
    public:
      typedef svm_held_pointer pointer_type;
      typedef size_t size_type;

    protected:
      std::shared_ptr<pyopencl::context> m_context;
      cl_uint m_alignment;
      cl_svm_mem_flags m_flags;
      pyopencl::command_queue_ref m_queue;

    public:
      svm_allocator(std::shared_ptr<pyopencl::context> const &ctx,
          cl_uint alignment, cl_svm_mem_flags flags=CL_MEM_READ_WRITE,
          pyopencl::command_queue *queue=nullptr)
        : m_context(ctx), m_alignment(alignment), m_flags(flags)
      {
        if (queue)
          m_queue.set(queue->data());
      }

      svm_allocator(svm_allocator const &src)
      : m_context(src.m_context), m_alignment(src.m_alignment),
      m_flags(src.m_flags)
      { }

      virtual ~svm_allocator()
      { }

      virtual bool is_deferred() const
      {
        // FIXME: I don't know whether that's true.
        return false;
      }

      std::shared_ptr<pyopencl::context> context() const
      {
        return m_context;
      }

      pointer_type allocate(size_type size)
      {
        if (size == 0)
          return { nullptr, nullptr };

        PYOPENCL_PRINT_CALL_TRACE("clSVMalloc");
        return {
          clSVMAlloc(m_context->data(), m_flags, size, m_alignment),
          pyopencl::command_queue_ref(m_queue.is_valid() ? m_queue.data() : nullptr)
        };
      }

      virtual pointer_type hand_out_existing_block(pointer_type &&p)
      {
        if (m_queue.is_valid())
        {
          if (p.queue.is_valid())
          {
            if (p.queue.data() != m_queue.data())
            {
              // make sure synchronization promises stay valid in new queue
              cl_event evt;

              PYOPENCL_CALL_GUARDED(clEnqueueMarker, (p.queue.data(), &evt));
              PYOPENCL_CALL_GUARDED(clEnqueueMarkerWithWaitList,
                  (m_queue.data(), 1, &evt, nullptr));
            }
          }
          p.queue.set(m_queue.data());
        }
        return std::move(p);
      }

      void free(pointer_type &&p)
      {
        if (p.queue.is_valid())
        {
          PYOPENCL_CALL_GUARDED_CLEANUP(clEnqueueSVMFree, (
                p.queue.data(), 1, &p.ptr,
                nullptr, nullptr,
                0, nullptr, nullptr));
          p.queue.reset();
        }
        else
        {
          PYOPENCL_PRINT_CALL_TRACE("clSVMFree");
          clSVMFree(m_context->data(), p.ptr);
        }
      }

      void try_release_blocks()
      {
        pyopencl::run_python_gc();
      }
  };

  // }}}


  // {{{ svm_allocator_call

  inline
  pyopencl::svm_allocation *svm_allocator_call(svm_allocator &alloc, size_t size)
  {
    int try_count = 0;
    while (true)
    {
      try
      {
        svm_held_pointer mem(alloc.allocate(size));
        if (mem.queue.is_valid())
          return new pyopencl::svm_allocation(
              alloc.context(), mem.ptr, size, mem.queue.data());
        else
          return new pyopencl::svm_allocation(
              alloc.context(), mem.ptr, size, nullptr);
      }
      catch (pyopencl::error &e)
      {
        if (!e.is_out_of_memory())
          throw;
        if (++try_count == 2)
          throw;
      }

      alloc.try_release_blocks();
    }
  }

  // }}}


  // {{{ pooled_svm

  class pooled_svm
    : public pyopencl::pooled_allocation<pyopencl::memory_pool<svm_allocator>>,
    public pyopencl::svm_pointer
  {
    private:
      typedef
        pyopencl::pooled_allocation<pyopencl::memory_pool<svm_allocator>>
        super;

    public:
      pooled_svm(
          std::shared_ptr<super::pool_type> p, super::size_type s)
        : super(p, s)
      { }

      void *svm_ptr() const
      { return m_ptr.ptr; }

      size_t size() const
      { return m_size; }

      void bind_to_queue(pyopencl::command_queue const &queue)
      {
        if (pyopencl::is_queue_out_of_order(queue.data()))
          throw pyopencl::error("PooledSVM.bind_to_queue", CL_INVALID_VALUE,
              "supplying an out-of-order queue to SVMAllocation is invalid");

        if (m_ptr.queue.is_valid())
        {
          // make sure synchronization promises stay valid in new queue
          cl_event evt;

          PYOPENCL_CALL_GUARDED(clEnqueueMarker, (m_ptr.queue.data(), &evt));
          PYOPENCL_CALL_GUARDED(clEnqueueWaitForEvents, (queue.data(), 1, &evt));
        }

        m_ptr.queue.set(queue.data());
      }

      void unbind_from_queue()
      {
        // NOTE: This absolves the allocation from any synchronization promises
        // made. Keeping those before calling this method is the responsibility
        // of the user.
        m_ptr.queue.reset();
      }
  };

  // }}}


  // {{{ svm_pool_allocate

  pooled_svm *svm_pool_allocate(
      std::shared_ptr<pyopencl::memory_pool<svm_allocator> > pool,
      pyopencl::memory_pool<svm_allocator>::size_type sz)
  {
    return new pooled_svm(pool, sz);
  }

  // }}}

#endif

  template<class Wrapper>
  void expose_memory_pool(Wrapper &wrapper)
  {
    typedef typename Wrapper::type cls;
    wrapper
      .def_property_readonly("held_blocks", &cls::held_blocks)
      .def_property_readonly("active_blocks", &cls::active_blocks)
      .def_property_readonly("managed_bytes", &cls::managed_bytes)
      .def_property_readonly("active_bytes", &cls::active_bytes)
      .DEF_SIMPLE_METHOD(bin_number)
      .DEF_SIMPLE_METHOD(alloc_size)
      .DEF_SIMPLE_METHOD(free_held)
      .DEF_SIMPLE_METHOD(stop_holding)
      ;
  }
}




void pyopencl_expose_mempool(py::module &m)
{
  m.def("bitlog2", pyopencl::bitlog2);

  {
    typedef buffer_allocator_base cls;
    py::class_<cls, std::shared_ptr<cls>> wrapper(
        m, "_tools_AllocatorBase"/*, py::no_init */);
    wrapper
      .def("__call__", buffer_allocator_call)
      ;

  }

  {
    typedef pyopencl::memory_pool<test_allocator> cls;

    py::class_<cls, std::shared_ptr<cls>> wrapper( m, "_TestMemoryPool");
    wrapper
      .def(py::init([](unsigned leading_bits_in_bin_id)
            { return new cls(
                std::shared_ptr<test_allocator>(new test_allocator()),
                leading_bits_in_bin_id); }),
          py::arg("leading_bits_in_bin_id")=4
          )
      .def("allocate", [](std::shared_ptr<cls> pool, cls::size_type sz)
          {
            pool->allocate(sz);
            return py::none();
          })
      ;

    expose_memory_pool(wrapper);
  }

  {
    typedef deferred_buffer_allocator cls;
    py::class_<cls, buffer_allocator_base, std::shared_ptr<cls>> wrapper(
        m, "_tools_DeferredAllocator");
    wrapper
      .def(py::init<
          std::shared_ptr<pyopencl::context> const &>())
      .def(py::init<
          std::shared_ptr<pyopencl::context> const &,
          cl_mem_flags>(),
          py::arg("queue"), py::arg("mem_flags"))
      ;
  }

  {
    typedef immediate_buffer_allocator cls;
    py::class_<cls, buffer_allocator_base, std::shared_ptr<cls>> wrapper(
        m, "_tools_ImmediateAllocator");
    wrapper
      .def(py::init<pyopencl::command_queue &>())
      .def(py::init<pyopencl::command_queue &, cl_mem_flags>(),
          py::arg("queue"), py::arg("mem_flags"))
      ;
  }

  {
    typedef pyopencl::memory_pool<buffer_allocator_base> cls;

    py::class_<
      cls, /* boost::noncopyable, */
      std::shared_ptr<cls>> wrapper( m, "_tools_MemoryPool");
    wrapper
      .def(py::init<std::shared_ptr<buffer_allocator_base>, unsigned>(),
          py::arg("allocator"),
          py::arg("leading_bits_in_bin_id")=4
          )
      .def("allocate", buffer_pool_allocate)
      .def("__call__", buffer_pool_allocate)
      // undoc for now
      .DEF_SIMPLE_METHOD(set_trace)
      ;

    expose_memory_pool(wrapper);
  }

  {
    typedef pooled_buffer cls;
    py::class_<cls, /* boost::noncopyable, */
      pyopencl::memory_object_holder>(
          m, "_tools_PooledBuffer"/* , py::no_init */)
      .def("release", &cls::free)
      // undocumented for now, for consistency with SVM
      .def("bind_to_queue", [](cls &self, pyopencl::command_queue &queue) { /* no-op */ })
      .def("unbind_from_queue", [](cls &self) { /* no-op */ })
      ;
  }

#if PYOPENCL_CL_VERSION >= 0x2000
  {
    typedef svm_allocator cls;
    py::class_<cls, std::shared_ptr<cls>> wrapper(
        m, "_tools_SVMAllocator");
    wrapper
      .def(py::init<std::shared_ptr<pyopencl::context>  const &, cl_uint, cl_uint, pyopencl::command_queue *>(),
          py::arg("context"),
          py::arg("alignment"),
          py::arg("flags")=CL_MEM_READ_WRITE,
          py::arg("queue").none(true)=nullptr
          )
      .def("__call__", svm_allocator_call)
      ;
  }

  {
    typedef pyopencl::memory_pool<svm_allocator> cls;

    py::class_<
      cls, /* boost::noncopyable, */
      std::shared_ptr<cls>> wrapper( m, "_tools_SVMPool");
    wrapper
      .def(py::init<std::shared_ptr<svm_allocator>, unsigned>(),
          py::arg("allocator"),
          py::arg("leading_bits_in_bin_id")=4
          )
      .def("allocate", svm_pool_allocate)
      .def("__call__", svm_pool_allocate)
      // undoc for now
      .DEF_SIMPLE_METHOD(set_trace)
      ;

    expose_memory_pool(wrapper);
  }

  {
    typedef pooled_svm cls;
    py::class_<cls, /* boost::noncopyable, */
      pyopencl::svm_pointer>(
          m, "_tools_PooledSVM"/* , py::no_init */)
      .def("release", &cls::free)
      .def("__eq__", [](const cls &self, const cls &other)
          { return self.svm_ptr() == other.svm_ptr(); })
      .def("__hash__", [](cls &self) { return (intptr_t) self.svm_ptr(); })
      .DEF_SIMPLE_METHOD(bind_to_queue)
      .DEF_SIMPLE_METHOD(unbind_from_queue)
      ;
  }

#endif
}

// vim: foldmethod=marker
