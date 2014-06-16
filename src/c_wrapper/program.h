#include "clhelper.h"

#ifndef __PYOPENCL_PROGRAM_H
#define __PYOPENCL_PROGRAM_H

namespace pyopencl {

class device;

// {{{ program

extern template class clobj<cl_program>;

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
            pyopencl_call_guarded(clRetainProgram, this);
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
        return pyopencl_get_vec_info(cl_device_id, Program, this,
                                     CL_PROGRAM_DEVICES);
    }
    generic_info get_info(cl_uint param_name) const;
    PYOPENCL_USE_RESULT generic_info
    get_build_info(const device *dev, cl_program_build_info param_name) const;

    // #if PYOPENCL_CL_VERSION >= 0x1020
    //       void compile(std::string options, py::object py_devices,
    //           py::object py_headers)
    //       {
    //         PYOPENCL_PARSE_PY_DEVICES;

    //         // {{{ pick apart py_headers
    //         // py_headers is a list of tuples *(name, program)*

    //         std::vector<std::string> header_names;
    //         std::vector<cl_program> programs;
    //         PYTHON_FOREACH(name_hdr_tup, py_headers)
    //         {
    //           if (py::len(name_hdr_tup) != 2)
    //             throw error("Program.compile", CL_INVALID_VALUE,
    //                 "epxected (name, header) tuple in headers list");
    //           std::string name = py::extract<std::string const &>(name_hdr_tup[0]);
    //           program &prg = py::extract<program &>(name_hdr_tup[1]);

    //           header_names.push_back(name);
    //           programs.push_back(prg.data());
    //         }

    //         std::vector<const char *> header_name_ptrs;
    //         BOOST_FOREACH(std::string const &name, header_names)
    //           header_name_ptrs.push_back(name.c_str());

    //         // }}}

    //         PYOPENCL_CALL_GUARDED(clCompileProgram,
    //             (this, num_devices, devices,
    //              options.c_str(), header_names.size(),
    //              programs.empty() ? nullptr : &programs.front(),
    //              header_name_ptrs.empty() ? nullptr : &header_name_ptrs.front(),
    //              0, 0));
    //       }
    // #endif
};

// }}}

}

#endif
