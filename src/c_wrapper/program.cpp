#include "program.h"
#include "device.h"
#include "context.h"
#include "clhelper.h"

namespace pyopencl {

template class clobj<cl_program>;

program::~program()
{
    pyopencl_call_guarded_cleanup(clReleaseProgram, this);
}

generic_info
program::get_info(cl_uint param) const
{
    switch ((cl_program_info)param) {
    case CL_PROGRAM_CONTEXT:
        return pyopencl_get_opaque_info(context, Program, this, param);
    case CL_PROGRAM_REFERENCE_COUNT:
    case CL_PROGRAM_NUM_DEVICES:
        return pyopencl_get_int_info(cl_uint, Program, this, param);
    case CL_PROGRAM_DEVICES:
        return pyopencl_get_opaque_array_info(device, Program, this, param);
    case CL_PROGRAM_SOURCE:
        return pyopencl_get_str_info(Program, this, param);
    case CL_PROGRAM_BINARY_SIZES:
        return pyopencl_get_array_info(size_t, Program, this, param);
    case CL_PROGRAM_BINARIES: {
        auto sizes = pyopencl_get_vec_info(size_t, Program, this,
                                           CL_PROGRAM_BINARY_SIZES);
        pyopencl_buf<char*> result_ptrs(sizes.len());
        for (size_t i  = 0;i < sizes.len();i++) {
            result_ptrs[i] = (char*)malloc(sizes[i]);
        }
        try {
            pyopencl_call_guarded(clGetProgramInfo, this, CL_PROGRAM_BINARIES,
                                  sizes.len() * sizeof(char*),
                                  result_ptrs.get(), nullptr);
        } catch (...) {
            for (size_t i  = 0;i < sizes.len();i++) {
                free(result_ptrs[i]);
            }
        }
        pyopencl_buf<generic_info> gis(sizes.len());
        for (size_t i  = 0;i < sizes.len();i++) {
            gis[i].value = result_ptrs[i];
            gis[i].dontfree = 0;
            gis[i].opaque_class = CLASS_NONE;
            gis[i].type =  _copy_str(std::string("char[") +
                                     tostring(sizes[i]) + "]");
        }
        return pyopencl_convert_array_info(generic_info, gis);
    }

#if PYOPENCL_CL_VERSION >= 0x1020
    case CL_PROGRAM_NUM_KERNELS:
        return pyopencl_get_int_info(size_t, Program, this, param);
    case CL_PROGRAM_KERNEL_NAMES:
        return pyopencl_get_str_info(Program, this, param);
#endif
    default:
        throw clerror("Program.get_info", CL_INVALID_VALUE);
    }
}

generic_info
program::get_build_info(const device *dev, cl_program_build_info param) const
{
    switch (param) {
    case CL_PROGRAM_BUILD_STATUS:
        return pyopencl_get_int_info(cl_build_status, ProgramBuild,
                                     this, dev, param);
    case CL_PROGRAM_BUILD_OPTIONS:
    case CL_PROGRAM_BUILD_LOG:
        return pyopencl_get_str_info(ProgramBuild, this, dev, param);
#if PYOPENCL_CL_VERSION >= 0x1020
    case CL_PROGRAM_BINARY_TYPE:
        return pyopencl_get_int_info(cl_program_binary_type, ProgramBuild,
                                     this, dev, param);
#endif
    default:
        throw clerror("Program.get_build_info", CL_INVALID_VALUE);
    }
}

}

// c wrapper
// Import all the names in pyopencl namespace for c wrappers.
using namespace pyopencl;

// Program
error*
create_program_with_source(clobj_t *prog, clobj_t _ctx, const char *src)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            size_t length = strlen(src);
            cl_program result = pyopencl_call_guarded(
                clCreateProgramWithSource, ctx, 1, &src, &length);
            *prog = new_program(result, KND_SOURCE);
        });
}

error*
create_program_with_binary(clobj_t *prog, clobj_t _ctx,
                           cl_uint num_devices, const clobj_t *devices,
                           char **binaries, size_t *binary_sizes)
{
    auto ctx = static_cast<context*>(_ctx);
    const auto devs = buf_from_class<device>(devices, num_devices);
    pyopencl_buf<cl_int> binary_statuses(num_devices);
    return c_handle_error([&] {
            cl_program result = pyopencl_call_guarded(
                clCreateProgramWithBinary, ctx, devs,
                binary_sizes, reinterpret_cast<const unsigned char**>(
                    const_cast<const char**>(binaries)), binary_statuses.get());
            // for (cl_uint i = 0; i < num_devices; ++i)
            //   std::cout << i << ":" << binary_statuses[i] << std::endl;
            *prog = new_program(result, KND_BINARY);
        });
}

error*
program__build(clobj_t _prog, const char *options,
               cl_uint num_devices, const clobj_t *_devices)
{
    auto prog = static_cast<const program*>(_prog);
    const auto devices = buf_from_class<device>(_devices, num_devices);
    return c_handle_error([&] {
            pyopencl_call_guarded(clBuildProgram, prog, devices, options,
                                  nullptr, nullptr);
        });
}

error*
program__kind(clobj_t prog, int *kind)
{
    return c_handle_error([&] {
            *kind = static_cast<program*>(prog)->kind();
        });
}

error*
program__get_build_info(clobj_t _prog, clobj_t _dev,
                        cl_program_build_info param, generic_info *out)
{
    auto prog = static_cast<program*>(_prog);
    auto dev = static_cast<device*>(_dev);
    return c_handle_error([&] {
            *out = prog->get_build_info(dev, param);
        });
}
