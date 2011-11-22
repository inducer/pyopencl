CLUDA_PREAMBLE = """
#define local_barrier() barrier(CLK_LOCAL_MEM_FENCE);

#define WITHIN_KERNEL /* empty */
#define KERNEL __kernel
#define GLOBAL_MEM __global
#define LOCAL_MEM __local
#define LOCAL_MEM_ARG __local
#define REQD_WG_SIZE(X,Y,Z) __attribute__((reqd_work_group_size(X, Y, Z)))

#define LID_0 get_local_id(0)
#define LID_1 get_local_id(1)
#define LID_2 get_local_id(2)

#define GID_0 get_group_id(0)
#define GID_1 get_group_id(1)
#define GID_2 get_group_id(2)

#define LDIM_0 get_local_size(0)
#define LDIM_1 get_local_size(1)
#define LDIM_2 get_local_size(2)

#define GDIM_0 get_num_groups(0)
#define GDIM_1 get_num_groups(1)
#define GDIM_2 get_num_groups(2)

% if double_support:
    #pragma OPENCL EXTENSION cl_khr_fp64: enable
% endif
"""




