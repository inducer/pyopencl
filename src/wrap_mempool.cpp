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



namespace pyopencl {
  // {{{ test_allocator

  class test_allocator
  {
    public:
      typedef void *pointer_type;
      typedef size_t size_type;

      bool is_deferred() const
      {
        return false;
      }

      pointer_type allocate(size_type s)
      {
        return nullptr;
      }

      pointer_type hand_out_existing_block(pointer_type &&p)
      {
        return p;
      }

      ~test_allocator()
      { }

      void free(pointer_type &&p)
      { }

      void try_release_blocks()
      { }
  };

  // }}}


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

      pointer_type hand_out_existing_block(pointer_type &&p)
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

      virtual ~pooled_buffer()
      { }

      const super::pointer_type data() const
      { return m_ptr; }

      size_t size() const
      {
        return m_size;
      }
  };

  // }}}


  // {{{ allocate_from_buffer_allocator

  inline
  buffer *allocate_from_buffer_allocator(buffer_allocator_base &alloc, size_t size)
  {
    cl_mem mem = nullptr;
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


  // {{{ allocate_from_buffer_pool

  pooled_buffer *allocate_from_buffer_pool(
      std::shared_ptr<memory_pool<buffer_allocator_base> > pool,
      memory_pool<buffer_allocator_base>::size_type sz)
  {
    return new pooled_buffer(pool, sz);
  }

  // }}}


#if PYOPENCL_CL_VERSION >= 0x2000

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
          cl_uint alignment=0, cl_svm_mem_flags flags=CL_MEM_READ_WRITE,
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

      ~svm_allocator()
      { }

      bool is_deferred() const
      {
        // According to experiments with the Nvidia implementation (and based
        // on my reading of the CL spec), clSVMalloc will return an error
        // immediately upon being out of memory.  Therefore the
        // immediate/deferred split on the buffer side is not needed here.
        // -AK, 2022-09-07

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

      pointer_type hand_out_existing_block(pointer_type &&p)
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
        else
        {
          if (p.queue.is_valid())
          {
            PYOPENCL_CALL_GUARDED_THREADED(clFinish, (p.queue.data()));
            p.queue.reset();
          }
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

      virtual ~pooled_svm()
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
          if (m_ptr.queue.data() != queue.data())
          {
            // make sure synchronization promises stay valid in new queue
            cl_event evt;

            PYOPENCL_CALL_GUARDED(clEnqueueMarker, (m_ptr.queue.data(), &evt));
            PYOPENCL_CALL_GUARDED(clEnqueueMarkerWithWaitList,
                (queue.data(), 1, &evt, nullptr));
          }
        }

        m_ptr.queue.set(queue.data());
      }

      void unbind_from_queue()
      {
        if (m_ptr.queue.is_valid())
          PYOPENCL_CALL_GUARDED_THREADED(clFinish, (m_ptr.queue.data()));

        m_ptr.queue.reset();
      }

      // only use for testing/diagnostic/debugging purposes!
      cl_command_queue queue() const
      {
        if (m_ptr.queue.is_valid())
          return m_ptr.queue.data();
        else
          return nullptr;
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


  // {{{ allocate_from_svm_pool

  pooled_svm *allocate_from_svm_pool(
      std::shared_ptr<pyopencl::memory_pool<svm_allocator> > pool,
      pyopencl::memory_pool<svm_allocator>::size_type sz)
  {
    return new pooled_svm(pool, sz);
  }

  // }}}

#endif
}


namespace {
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

      // undoc for now
      .def("_set_trace", &cls::set_trace)
      ;
  }
}




void pyopencl_expose_mempool(py::module_ &m)
{
  m.def("bitlog2", pyopencl::bitlog2);

  {
    typedef pyopencl::buffer_allocator_base cls;
    py::class_<cls, std::shared_ptr<cls>> wrapper(m, "AllocatorBase");
    wrapper
      .def("__call__", pyopencl::allocate_from_buffer_allocator, py::arg("size"))
      ;

  }

  {
    typedef pyopencl::memory_pool<pyopencl::test_allocator> cls;

    py::class_<cls, std::shared_ptr<cls>> wrapper( m, "_TestMemoryPool");
    wrapper
      .def(py::init([](unsigned leading_bits_in_bin_id)
            { return new cls(
                std::shared_ptr<pyopencl::test_allocator>(
                  new pyopencl::test_allocator()),
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
    typedef pyopencl::deferred_buffer_allocator cls;
    py::class_<cls, pyopencl::buffer_allocator_base, std::shared_ptr<cls>> wrapper(
        m, "DeferredAllocator");
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
    typedef pyopencl::immediate_buffer_allocator cls;
    py::class_<cls, pyopencl::buffer_allocator_base, std::shared_ptr<cls>> wrapper(
        m, "ImmediateAllocator");
    wrapper
      .def(py::init<pyopencl::command_queue &>())
      .def(py::init<pyopencl::command_queue &, cl_mem_flags>(),
          py::arg("queue"), py::arg("mem_flags"))
      ;
  }

  {
    typedef pyopencl::pooled_buffer cls;
    py::class_<cls, pyopencl::memory_object_holder>(m, "PooledBuffer")
      .def("release", &cls::free)

      .def("bind_to_queue", [](cls &self, pyopencl::command_queue &queue) { /* no-op */ })
      .def("unbind_from_queue", [](cls &self) { /* no-op */ })
      ;
  }

  {
    typedef pyopencl::memory_pool<pyopencl::buffer_allocator_base> cls;

    py::class_<cls, std::shared_ptr<cls>> wrapper( m, "MemoryPool");
    wrapper
      .def(py::init<std::shared_ptr<pyopencl::buffer_allocator_base>, unsigned>(),
          py::arg("allocator"),
          py::arg("leading_bits_in_bin_id")=4
          )
      .def("allocate", pyopencl::allocate_from_buffer_pool, py::arg("size"))
      .def("__call__", pyopencl::allocate_from_buffer_pool, py::arg("size"))
      ;

    expose_memory_pool(wrapper);
  }

#if PYOPENCL_CL_VERSION >= 0x2000
  {
    typedef pyopencl::svm_allocator cls;
    py::class_<cls, std::shared_ptr<cls>> wrapper(m, "SVMAllocator");
    wrapper
      .def(py::init<std::shared_ptr<pyopencl::context>  const &, cl_uint, cl_uint, pyopencl::command_queue *>(),
          py::arg("context"),
          py::kw_only(),
          py::arg("alignment")=0,
          py::arg("flags")=CL_MEM_READ_WRITE,
          py::arg("queue").none(true)=nullptr
          )
      .def("__call__", pyopencl::svm_allocator_call, py::arg("size"))
      ;
  }

  {
    typedef pyopencl::pooled_svm cls;
    py::class_<cls, pyopencl::svm_pointer>(m, "PooledSVM")
      .def("release", &cls::free)
      .def("enqueue_release", &cls::free)
      .def("__eq__", [](const cls &self, const cls &other)
          { return self.svm_ptr() == other.svm_ptr(); })
      .def("__hash__", [](cls &self) { return (intptr_t) self.svm_ptr(); })
      .DEF_SIMPLE_METHOD(bind_to_queue)
      .DEF_SIMPLE_METHOD(unbind_from_queue)

      // only for diagnostic/debugging/testing purposes!
      .def_property_readonly("_queue",
          [](cls const &self) -> py::object
          {
            cl_command_queue queue = self.queue();
            if (queue)
              return py::cast(new pyopencl::command_queue(queue, true));
            else
              return py::none();
          })
      ;
  }

  {
    typedef pyopencl::memory_pool<pyopencl::svm_allocator> cls;

    py::class_<cls, std::shared_ptr<cls>> wrapper( m, "SVMPool");
    wrapper
      .def(py::init<std::shared_ptr<pyopencl::svm_allocator>, unsigned>(),
          py::arg("allocator"),
          py::kw_only(),
          py::arg("leading_bits_in_bin_id")=4
          )
      .def("__call__", pyopencl::allocate_from_svm_pool, py::arg("size"))
      ;

    expose_memory_pool(wrapper);
  }

#endif
}

// vim: foldmethod=marker
