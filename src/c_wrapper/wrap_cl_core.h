typedef enum {
  generic_info_type_cl_uint,
  generic_info_type_cl_mem_object_type,
} generic_info_type_t;
typedef struct {
  generic_info_type_t type;
  union value_t {
    cl_uint _cl_uint;
    cl_mem_object_type _cl_mem_object_type;
  } value;
} generic_info;

typedef struct {
  const char* type;
  const char* name;
  unsigned int value;
} constant;

int get_cl_version(void);
void* get_platforms(void** ptr_platforms, uint32_t* num_platforms);
void* platform__get_info(void* ptr_platform, cl_platform_info param_name, char** out);
void* platform__get_devices(void* ptr_platform, void** ptr_devices, uint32_t* num_devices, cl_device_type devtype);
void* device__get_info(void* ptr_device, cl_device_info param_name, char** out);
long device__hash(void *ptr_device);
void* _create_context(void** ptr_ctx, cl_context_properties* properties, cl_uint num_devices, void** ptr_devices);
void* _create_command_queue(void** ptr_command_queue, void* ptr_context, void* ptr_device, cl_command_queue_properties properties);
void* _create_buffer(void** ptr_buffer, void* ptr_context, cl_mem_flags flags, size_t size, void* hostbuf);
void* _create_program_with_source(void **ptr_program, void *ptr_context, char* src);
void* _create_program_with_binary(void **ptr_program, void *ptr_context, cl_uint num_devices, void** ptr_devices, cl_uint num_binaries, char** binaries);
void* program__build(void* ptr_program, char* options, cl_uint num_devices, void** ptr_devices);
void* program__kind(void* ptr_program, int *kind);
void* program__get_info__devices(void* ptr_program, void** ptr_devices, uint32_t* num_devices);
void* program__get_info__binaries(void* ptr_program, char*** ptr_binaries, uint32_t* num_binaries);
void* _create_kernel(void** ptr_kernel, void* ptr_program, char* name);
void* kernel__get_info(void *ptr_kernel, cl_kernel_info param, generic_info* out);
void* kernel__set_arg_mem_buffer(void* ptr_kernel, cl_uint arg_index, void* ptr_buffer);
void* _enqueue_nd_range_kernel(void **ptr_event, void* ptr_command_queue, void* ptr_kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size);
void* _enqueue_read_buffer(void **ptr_event, void* ptr_command_queue, void* ptr_memory_object_holder, void* buffer, size_t size, size_t device_offset, int is_blocking);
void* memory_object_holder__get_info(void* ptr_memory_object_holder, cl_mem_info param, generic_info* out);
void get_constants(constant** out, uint32_t *num_constants, void(*)(const char*));
void populate_constants(void(*add)(const char*, const char*, unsigned int value));
void freem(void*);









