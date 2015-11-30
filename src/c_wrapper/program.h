#include "clhelper.h"

#ifndef __PYOPENCL_PROGRAM_H
#define __PYOPENCL_PROGRAM_H

class device;

// {{{ program

extern template class clobj<cl_program>;
extern template void print_arg<cl_program>(std::ostream&,
                                           const cl_program&, bool);
extern template void print_buf<cl_program>(std::ostream&, const cl_program*,
                                           size_t, ArgType, bool, bool);

class program : public clobj<cl_program> {
private:
    program_kind_type m_program_kind;

public:
    PYOPENCL_DEF_CL_CLASS(PROGRAM);
    PYOPENCL_INLINE
    program(cl_program prog, bool retain,
            program_kind_type progkind=KND_UNKNOWN)
        : clobj(prog), m_program_kind(progkind)
    {
        if (retain) {
            pyopencl_call_guarded(clRetainProgram, PYOPENCL_CL_CASTABLE_THIS);
        }
    }
    ~program();
    PYOPENCL_USE_RESULT PYOPENCL_INLINE program_kind_type
    kind() const
    {
        return m_program_kind;
    }
    PYOPENCL_USE_RESULT pyopencl_buf<cl_device_id>
    get_info__devices() const
    {
        return pyopencl_get_vec_info(cl_device_id, Program, PYOPENCL_CL_CASTABLE_THIS,
                                     CL_PROGRAM_DEVICES);
    }
    generic_info get_info(cl_uint param_name) const;
    PYOPENCL_USE_RESULT generic_info
    get_build_info(const device *dev, cl_program_build_info param_name) const;
#if PYOPENCL_CL_VERSION >= 0x1020
    void compile(const char *opts, const clobj_t *_devs, size_t num_devs,
                 const clobj_t *_prgs, const char *const *names,
                 size_t num_hdrs);
#endif
    pyopencl_buf<clobj_t> all_kernels();
};

extern template void print_clobj<program>(std::ostream&, const program*);

// }}}

#endif
