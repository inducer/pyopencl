typedef enum { KND_UNKNOWN, KND_SOURCE, KND_BINARY } program_kind_type;

typedef struct {
  const char *routine;
  const char *msg;
  cl_int code;
  int other;
} error;

typedef enum {
  CLASS_NONE,
  CLASS_PLATFORM,
  CLASS_DEVICE,
  CLASS_KERNEL,
  CLASS_CONTEXT,
  CLASS_BUFFER,
  CLASS_PROGRAM,
  CLASS_EVENT,
  CLASS_COMMAND_QUEUE,
  CLASS_GL_BUFFER,
  CLASS_GL_RENDERBUFFER
} class_t;


typedef struct {
  class_t opaque_class;
  const char *type;
  void *value;
  int dontfree;
} generic_info;


int get_cl_version(void);
error *get_platforms(void **ptr_platforms, uint32_t *num_platforms);
error *platform__get_devices(void *ptr_platform, void **ptr_devices, uint32_t *num_devices, cl_device_type devtype);
error *_create_context(void **ptr_ctx, cl_context_properties *properties, cl_uint num_devices, void **ptr_devices);
error *_create_command_queue(void **ptr_command_queue, void *ptr_context, void *ptr_device, cl_command_queue_properties properties);
error *_create_buffer(void **ptr_buffer, void *ptr_context, cl_mem_flags flags, size_t size, void *hostbuf);
error *_create_program_with_source(void **ptr_program, void *ptr_context, char *src);
error *_create_program_with_binary(void **ptr_program, void *ptr_context, cl_uint num_devices, void **ptr_devices, cl_uint num_binaries, char **binaries, size_t *binary_sizes);
error *program__build(void *ptr_program, char *options, cl_uint num_devices, void **ptr_devices);
error *program__kind(void *ptr_program, int *kind);
error *program__get_build_info(void *ptr_program, void *ptr_device, cl_program_build_info param, generic_info *out);

error *event__get_profiling_info(void *ptr_event, cl_profiling_info param, generic_info *out);
error *event__wait(void *ptr_event);

error *_create_kernel(void **ptr_kernel, void *ptr_program, char *name);
error *kernel__set_arg_mem_buffer(void *ptr_kernel, cl_uint arg_index, void *ptr_buffer);
error *kernel__get_work_group_info(void *ptr_kernel, cl_kernel_work_group_info param, void *ptr_device, generic_info *out);
long _hash(void *ptr_platform, class_t);

error *_enqueue_nd_range_kernel(void **ptr_event, void *ptr_command_queue, void *ptr_kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size);
error *_enqueue_read_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_memory_object_holder, void *buffer, size_t size, size_t device_offset, void **wait_for, uint32_t num_wait_for, int is_blocking);
error *_enqueue_copy_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_src, void *ptr_dst, ptrdiff_t byte_count, size_t src_offset, size_t dst_offset, void **wait_for, uint32_t num_wait_for);
error *_enqueue_write_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_memory_object_holder, void *buffer, size_t size, size_t device_offset, void **wait_for, uint32_t num_wait_for, int is_blocking);
void populate_constants(void(*add)(const char*, const char*, long value));

intptr_t _int_ptr(void*, class_t);
void* _from_int_ptr(void **ptr_out, intptr_t int_ptr_value, class_t);
error *_get_info(void *ptr, class_t class_, cl_uint param, generic_info *out);
void _delete(void *ptr, class_t class_);
void _free(void*);
void _free2(void**, uint32_t size);

unsigned bitlog2(unsigned long v);

/* gl interop */

int have_gl();
error *_create_from_gl_buffer(void **ptr, void *ptr_context, cl_mem_flags flags, GLuint bufobj);
error *_create_from_gl_renderbuffer(void **ptr, void *ptr_context, cl_mem_flags flags, GLuint bufobj);
error *_enqueue_acquire_gl_objects(void **ptr_event, void *ptr_command_queue, void **ptr_mem_objects, uint32_t num_mem_objects, void **wait_for, uint32_t num_wait_for);
error *_enqueue_release_gl_objects(void **ptr_event, void *ptr_command_queue, void **ptr_mem_objects, uint32_t num_mem_objects, void **wait_for, uint32_t num_wait_for);
