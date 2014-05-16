// Everything in here should have a 'pyopencl_' prefix to avoid clashing with
// other libraries imported via CFFI.

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
  CLASS_GL_RENDERBUFFER,
  CLASS_IMAGE,
  CLASS_SAMPLER
} class_t;


typedef struct {
  class_t opaque_class;
  const char *type;
  void *value;
  int dontfree;
} generic_info;


int pyopencl_get_cl_version(void);

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

error *_create_sampler(void **ptr_sampler, void *ptr_context, int normalized_coordinates, cl_addressing_mode am, cl_filter_mode fm);

error *event__get_profiling_info(void *ptr_event, cl_profiling_info param, generic_info *out);
error *event__wait(void *ptr_event);

error *_create_kernel(void **ptr_kernel, void *ptr_program, char *name);
error *kernel__set_arg_null(void *ptr_kernel, cl_uint arg_index);
error *kernel__set_arg_mem(void *ptr_kernel, cl_uint arg_index, void *ptr_mem);
error *kernel__set_arg_sampler(void *ptr_kernel, cl_uint arg_index, void *ptr_sampler);
error *kernel__set_arg_buf(void *ptr_kernel, cl_uint arg_index, void *buffer, size_t size);

error *kernel__get_work_group_info(void *ptr_kernel, cl_kernel_work_group_info param, void *ptr_device, generic_info *out);

error *_get_supported_image_formats(void *ptr_context, cl_mem_flags flags, cl_mem_object_type image_type, generic_info *out);

error *_create_image_2d(void **ptr_image, void *ptr_context, cl_mem_flags flags, cl_image_format *fmt, size_t width, size_t height, size_t pitch, void *ptr_buffer, size_t size);
error *_create_image_3d(void **ptr_image, void *ptr_context, cl_mem_flags flags, cl_image_format *fmt, size_t width, size_t height, size_t depth, size_t pitch_x, size_t pitch_y, void *ptr_buffer, size_t size);
error *image__get_image_info(void *ptr_image, cl_image_info param, generic_info *out);

long _hash(void *ptr_platform, class_t);

error *_enqueue_nd_range_kernel(void **ptr_event, void *ptr_command_queue, void *ptr_kernel, cl_uint work_dim, const size_t *global_work_offset, const size_t *global_work_size, const size_t *local_work_size, void **wait_for, uint32_t num_wait_for);

error *_enqueue_marker_with_wait_list(void **ptr_event, void *ptr_command_queue,
                                      void **wait_for, uint32_t num_wait_for);
error *_enqueue_barrier_with_wait_list(void **ptr_event,
                                       void *ptr_command_queue,
                                       void **wait_for, uint32_t num_wait_for);
error *_enqueue_marker(void **ptr_event, void *ptr_command_queue);
error *_enqueue_barrier(void *ptr_command_queue);
error *_enqueue_read_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_mem, void *buffer, size_t size, size_t device_offset, void **wait_for, uint32_t num_wait_for, int is_blocking);
error *_enqueue_copy_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_src, void *ptr_dst, ptrdiff_t byte_count, size_t src_offset, size_t dst_offset, void **wait_for, uint32_t num_wait_for);
error *_enqueue_write_buffer(void **ptr_event, void *ptr_command_queue, void *ptr_memory_object_holder, void *buffer, size_t size, size_t device_offset, void **wait_for, uint32_t num_wait_for, int is_blocking);
error *_enqueue_read_image(void **ptr_event, void *ptr_command_queue, void *ptr_mem, size_t *origin, size_t *region, void *buffer, size_t size, size_t row_pitch, size_t slice_pitch, void **wait_for, uint32_t num_wait_for, int is_blocking);
void populate_constants(void(*add)(const char*, const char*, long value));

intptr_t _int_ptr(void*, class_t);
void* _from_int_ptr(void **ptr_out, intptr_t int_ptr_value, class_t);
error *_get_info(void *ptr, class_t class_, cl_uint param, generic_info *out);
void _delete(void *ptr, class_t class_);

void pyopencl_free_pointer(void*);
void pyopencl_free_pointer_array(void**, uint32_t size);

int pyopencl_have_gl();

unsigned pyopencl_bitlog2(unsigned long v);
