#include "program.h"
#include "device.h"
#include "context.h"
#include "clhelper.h"
#include "kernel.h"

template class clobj<cl_program>;
template void print_arg<cl_program>(std::ostream&, const cl_program&, bool);
template void print_clobj<program>(std::ostream&, const program*);
template void print_buf<cl_program>(std::ostream&, const cl_program*,
                                    size_t, ArgType, bool, bool);

PYOPENCL_USE_RESULT static PYOPENCL_INLINE program*
new_program(cl_program prog, program_kind_type progkind=KND_UNKNOWN)
{
    return pyopencl_convert_obj(program, clReleaseProgram, prog, progkind);
}

program::~program()
{
    pyopencl_call_guarded_cleanup(clReleaseProgram, PYOPENCL_CL_CASTABLE_THIS);
}

generic_info
program::get_info(cl_uint param) const
{
    switch ((cl_program_info)param) {
    case CL_PROGRAM_CONTEXT:
        return pyopencl_get_opaque_info(context, Program, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_PROGRAM_REFERENCE_COUNT:
    case CL_PROGRAM_NUM_DEVICES:
        return pyopencl_get_int_info(cl_uint, Program, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_PROGRAM_DEVICES:
        return pyopencl_get_opaque_array_info(device, Program, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_PROGRAM_SOURCE:
        return pyopencl_get_str_info(Program, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_PROGRAM_BINARY_SIZES:
        return pyopencl_get_array_info(size_t, Program, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_PROGRAM_BINARIES: {
        auto sizes = pyopencl_get_vec_info(size_t, Program, PYOPENCL_CL_CASTABLE_THIS,
                                           CL_PROGRAM_BINARY_SIZES);
        pyopencl_buf<char*> result_ptrs(sizes.len());
        for (size_t i  = 0;i < sizes.len();i++) {
            result_ptrs[i] = (char*)malloc(sizes[i]);
        }
        try {
            pyopencl_call_guarded(clGetProgramInfo, PYOPENCL_CL_CASTABLE_THIS, CL_PROGRAM_BINARIES,
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
        return pyopencl_get_int_info(size_t, Program, PYOPENCL_CL_CASTABLE_THIS, param);
    case CL_PROGRAM_KERNEL_NAMES:
        return pyopencl_get_str_info(Program, PYOPENCL_CL_CASTABLE_THIS, param);
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
                                     PYOPENCL_CL_CASTABLE_THIS, dev, param);
    case CL_PROGRAM_BUILD_OPTIONS:
    case CL_PROGRAM_BUILD_LOG:
        return pyopencl_get_str_info(ProgramBuild, PYOPENCL_CL_CASTABLE_THIS, dev, param);
#if PYOPENCL_CL_VERSION >= 0x1020
    case CL_PROGRAM_BINARY_TYPE:
        return pyopencl_get_int_info(cl_program_binary_type, ProgramBuild,
                                     PYOPENCL_CL_CASTABLE_THIS, dev, param);
#endif
#if PYOPENCL_CL_VERSION >= 0x2000
    case CL_PROGRAM_BUILD_GLOBAL_VARIABLE_TOTAL_SIZE:
        return pyopencl_get_int_info(size_t, ProgramBuild,
                                     PYOPENCL_CL_CASTABLE_THIS, dev, param);
#endif
    default:
        throw clerror("Program.get_build_info", CL_INVALID_VALUE);
    }
}

#if PYOPENCL_CL_VERSION >= 0x1020
void
program::compile(const char *opts, const clobj_t *_devs, size_t num_devs,
                 const clobj_t *_prgs, const char *const *names,
                 size_t num_hdrs)
{
    const auto devs = buf_from_class<device>(_devs, num_devs);
    const auto prgs = buf_from_class<program>(_prgs, num_hdrs);
    pyopencl_call_guarded(clCompileProgram, PYOPENCL_CL_CASTABLE_THIS, devs, opts, prgs,
                          buf_arg(names, num_hdrs), nullptr, nullptr);
}
#endif

pyopencl_buf<clobj_t>
program::all_kernels()
{
    cl_uint num_knls;
    pyopencl_call_guarded(clCreateKernelsInProgram, PYOPENCL_CL_CASTABLE_THIS, 0, nullptr,
                          buf_arg(num_knls));
    pyopencl_buf<cl_kernel> knls(num_knls);
    pyopencl_call_guarded(clCreateKernelsInProgram, PYOPENCL_CL_CASTABLE_THIS, knls,
                          buf_arg(num_knls));
    return buf_to_base<kernel>(knls, true);
}

// c wrapper

// Program
error*
create_program_with_source(clobj_t *prog, clobj_t _ctx, const char *_src)
{
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            const auto &src = _src;
            const size_t length = strlen(src);
            cl_program result = pyopencl_call_guarded(
                clCreateProgramWithSource, ctx, len_arg(src), buf_arg(length));
            *prog = new_program(result, KND_SOURCE);
        });
}

error*
create_program_with_il(clobj_t *prog, clobj_t _ctx, void *il, size_t length)
{
#if PYOPENCL_CL_VERSION >= 0x2010
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            cl_program result = pyopencl_call_guarded(
                clCreateProgramWithIL, ctx, il, length);
            *prog = new_program(result, KND_SOURCE);
        });
#else
    PYOPENCL_UNSUPPORTED_BEFORE(clCreateProgramWithIL, "CL 2.1")
#endif
}

error*
create_program_with_binary(clobj_t *prog, clobj_t _ctx,
                           cl_uint num_devices, const clobj_t *devices,
                           const unsigned char **binaries, size_t *binary_sizes)
{
    auto ctx = static_cast<context*>(_ctx);
    const auto devs = buf_from_class<device>(devices, num_devices);
    pyopencl_buf<cl_int> binary_statuses(num_devices);
    return c_handle_error([&] {
            cl_program result = pyopencl_call_guarded(
                clCreateProgramWithBinary, ctx, devs,
                binary_sizes, binaries, binary_statuses.get());
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

error*
program__create_with_builtin_kernels(clobj_t *_prg, clobj_t _ctx,
                                     const clobj_t *_devs, uint32_t num_devs,
                                     const char *names)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    const auto devs = buf_from_class<device>(_devs, num_devs);
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto prg = pyopencl_call_guarded(clCreateProgramWithBuiltInKernels,
                                             ctx, devs, names);
            *_prg = new_program(prg);
        });
#else
    PYOPENCL_UNSUPPORTED(clCreateProgramWithBuiltInKernels, "CL 1.1 and below")
#endif
}

error*
program__compile(clobj_t _prg, const char *opts, const clobj_t *_devs,
                 size_t num_devs, const clobj_t *_prgs,
                 const char *const *names, size_t num_hdrs)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    auto prg = static_cast<program*>(_prg);
    return c_handle_error([&] {
            prg->compile(opts, _devs, num_devs, _prgs, names, num_hdrs);
        });
#else
    PYOPENCL_UNSUPPORTED(clCompileProgram, "CL 1.1 and below")
#endif
}

error*
program__link(clobj_t *_prg, clobj_t _ctx, const clobj_t *_prgs,
              size_t num_prgs, const char *opts, const clobj_t *_devs,
              size_t num_devs)
{
#if PYOPENCL_CL_VERSION >= 0x1020
    const auto devs = buf_from_class<device>(_devs, num_devs);
    const auto prgs = buf_from_class<program>(_prgs, num_prgs);
    auto ctx = static_cast<context*>(_ctx);
    return c_handle_error([&] {
            auto prg = pyopencl_call_guarded(clLinkProgram, ctx, devs, opts,
                                             prgs, nullptr, nullptr);
            *_prg = new_program(prg);
        });
#else
    PYOPENCL_UNSUPPORTED(clLinkProgram, "CL 1.1 and below")
#endif
}

error*
program__all_kernels(clobj_t _prg, clobj_t **_knl, uint32_t *size)
{
    auto prg = static_cast<program*>(_prg);
    return c_handle_error([&] {
            auto knls = prg->all_kernels();
            *size = knls.len();
            *_knl = knls.release();
        });
}
