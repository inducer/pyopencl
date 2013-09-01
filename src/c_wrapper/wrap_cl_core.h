typedef enum { KND_UNKNOWN, KND_SOURCE, KND_BINARY } program_kind_type;

typedef struct {
  const char *type;
  void *value;
} generic_info;

typedef struct {
  const char *routine;
  const char *msg;
  cl_int code;
} error;

typedef enum {
  CLASS_PLATFORM,
  CLASS_DEVICE,
  CLASS_KERNEL,
  CLASS_CONTEXT,
  CLASS_BUFFER,
  CLASS_PROGRAM,
  CLASS_EVENT,
  CLASS_COMMAND_QUEUE
} class_t;

int get_cl_version(void);
error *get_platforms(void **ptr_platforms, uint32_t *num_platforms);
error *platform__get_info(void *ptr_platform, cl_platform_info param_name, generic_info *out);
error *platform__get_devices(void *ptr_platform, void **ptr_devices, uint32_t *num_devices, cl_device_type devtype);
error *device__get_info(void *ptr_device, cl_device_info param_name, generic_info *out);
error *_create_context(void **ptr_ctx, cl_context_properties *properties, cl_uint num_devices, void **ptr_devices);
error *_create_command_queue(void **ptr_command_queue, void *ptr_context, void *ptr_device, cl_command_queue_properties properties);
error *_create_buffer(void **ptr_buffer, void *ptr_context, cl_mem_flags flags, size_t size, void *hostbuf);
error *_create_program_with_source(void **ptr_program, void *ptr_context, char *src);
error *_create_program_with_binary(void **ptr_program, void *ptr_context, cl_uint num_devices, void **ptr_devices, cl_uint num_binaries, char **binaries);
error *program__build(void *ptr_program, char *options, cl_uint num_devices, void **ptr_devices);
error *program__kind(void *ptr_program, int *kind);
error *program__get_build_info(void *ptr_program, void *ptr_device, cl_program_build_info param, generic_info *out);
error *program__get_info(void *ptr_program, cl_program_info param, generic_info *out);
error *program__get_info__devices(void *ptr_program, void **ptr_devices, uint32_t *num_devices);
error *program__get_info__binaries(void *ptr_program, char ***ptr_binaries, uint32_t *num_binaries);

error *_create_kernel(void **ptr_kernel, void *ptr_program, char *name);
error *kernel__get_info(void *ptr_kernel, cl_kernel_info param, generic_info *out);
error *kernel__set_arg_mem_buffer(void *ptr_kernel, cl_uint arg_index, void *ptr_buffer);
long _hash(void *ptr_platform, class_t);

error *_enqueue_nd_range_kernel(void **ptr_event, void *ptr_command_queue, void *ptr_kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size);
error *_enqueue_read_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_memory_object_holder, void *buffer, size_t size, size_t device_offset, int is_blocking);
error *memory_object_holder__get_info(void *ptr_memory_object_holder, cl_mem_info param, generic_info *out);
void populate_constants(void(*add)(const char*, const char*, long value));

intptr_t _int_ptr(void*, class_t);
void* _from_int_ptr(void **ptr_out, intptr_t int_ptr_value, class_t);

void _free(void*);
void _free2(void**, uint32_t size);




