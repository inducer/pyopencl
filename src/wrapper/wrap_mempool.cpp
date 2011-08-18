// Gregor Thalhammer (on Apr 13, 2011) said it's necessary to import Python.h 
// first to prevent OS X from overriding a bunch of macros. (e.g. isspace)
#include <Python.h>

#include <vector>
#include "wrap_helpers.hpp"
#include "wrap_cl.hpp"
#include "mempool.hpp"
#include "tools.hpp"
#include <boost/python/stl_iterator.hpp>




namespace py = boost::python;




namespace
{
  class cl_allocator
  {
      boost::shared_ptr<pyopencl::context> m_context;
      cl_mem_flags m_flags;

    public:
      cl_allocator(boost::shared_ptr<pyopencl::context> const &ctx,
          cl_mem_flags flags=CL_MEM_READ_WRITE)
        : m_context(ctx), m_flags(flags)
      {
        if (flags & (CL_MEM_USE_HOST_PTR | CL_MEM_COPY_HOST_PTR))
          throw pyopencl::error("PoolAllocator", CL_INVALID_VALUE,
              "cannot specify USE_HOST_PTR or COPY_HOST_PTR flags");
      }

      typedef cl_mem pointer_type;
      typedef size_t size_type;

      pointer_type allocate(size_type s)
      {
        return pyopencl::create_buffer(m_context->data(), m_flags, s, 0);
      }

      void free(pointer_type p)
      {
        PYOPENCL_CALL_GUARDED(clReleaseMemObject, (p));
      }

      void try_release_blocks()
      {
        pyopencl::run_python_gc();
      }
  };




  inline
  pyopencl::buffer *allocator_call(cl_allocator &alloc, size_t size)
  {
    cl_mem mem = alloc.allocate(size);

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




  class pooled_buffer
    : public pyopencl::pooled_allocation<pyopencl::memory_pool<cl_allocator> >,
    public pyopencl::memory_object_holder
  {
    private:
      typedef
        pyopencl::pooled_allocation<pyopencl::memory_pool<cl_allocator> >
        super;

    public:
      pooled_buffer(
          boost::shared_ptr<super::pool_type> p, super::size_type s)
        : super(p, s)
      { }

      const super::pointer_type data() const
      { return ptr(); }
  };




  pooled_buffer *device_pool_allocate(
      boost::shared_ptr<pyopencl::memory_pool<cl_allocator> > pool,
      pyopencl::memory_pool<cl_allocator>::size_type sz)
  {
    return new pooled_buffer(pool, sz);
  }




  template<class Wrapper>
  void expose_memory_pool(Wrapper &wrapper)
  {
    typedef typename Wrapper::wrapped_type cls;
    wrapper
      .add_property("held_blocks", &cls::held_blocks)
      .add_property("active_blocks", &cls::active_blocks)
      .DEF_SIMPLE_METHOD(bin_number)
      .DEF_SIMPLE_METHOD(alloc_size)
      .DEF_SIMPLE_METHOD(free_held)
      .DEF_SIMPLE_METHOD(stop_holding)
      .staticmethod("bin_number")
      .staticmethod("alloc_size")
      ;
  }
}




void pyopencl_expose_mempool()
{
  py::def("bitlog2", pyopencl::bitlog2);

  {
    typedef cl_allocator cls;
    py::class_<cls> wrapper("CLAllocator",
        py::init<
          boost::shared_ptr<pyopencl::context> const &,
          py::optional<cl_mem_flags> >());
    wrapper
      .def("__call__", allocator_call,
          py::return_value_policy<py::manage_new_object>())
      ;

  }

  {
    typedef pyopencl::memory_pool<cl_allocator> cl;

    py::class_<
      cl, boost::noncopyable,
      boost::shared_ptr<cl> > wrapper("MemoryPool",
          py::init<cl_allocator const &>()
          );
    wrapper
      .def("allocate", device_pool_allocate,
          py::return_value_policy<py::manage_new_object>())
      .def("__call__", device_pool_allocate,
          py::return_value_policy<py::manage_new_object>())
      ;

    expose_memory_pool(wrapper);
  }

  {
    typedef pooled_buffer cls;
    py::class_<cls, boost::noncopyable, 
      py::bases<pyopencl::memory_object_holder> >(
        "PooledBuffer", py::no_init)
      .def("release", &cls::free)
      ;
  }
}
