// Interface between C and Python

// Types
typedef enum {
    TYPE_FLOAT,
    TYPE_INT,
    TYPE_UINT,
} type_t;

typedef enum {
    KND_UNKNOWN,
    KND_SOURCE,
    KND_BINARY
} program_kind_type;

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


// Generic functions
int get_cl_version();
void free_pointer(void*);
void free_pointer_array(void**, uint32_t size);
void set_py_funcs(int (*_gc)(), void *(*_ref)(void*), void (*_deref)(void*),
                  void (*_call)(void*, cl_int));
int have_gl();

unsigned bitlog2(unsigned long v);
void populate_constants(void(*add)(const char*, const char*, long value));
int get_debug();
void set_debug(int debug);

// Platform
error *get_platforms(clobj_t **ptr_platforms, uint32_t *num_platforms);
error *platform__get_devices(clobj_t platform, clobj_t **ptr_devices,
                             uint32_t *num_devices, cl_device_type devtype);
error *platform__unload_compiler(clobj_t plat);
// Device
error *device__create_sub_devices(clobj_t _dev, clobj_t **_devs,
                                  const cl_device_partition_property *props,
                                  uint32_t *num_devices);
// Context
error *create_context(clobj_t *ctx, const cl_context_properties *props,
                      cl_uint num_devices, const clobj_t *ptr_devices);
error *create_context_from_type(clobj_t *_ctx,
                                const cl_context_properties *props,
                                cl_device_type dev_type);
error *context__get_supported_image_formats(clobj_t context, cl_mem_flags flags,
                                            cl_mem_object_type image_type,
                                            generic_info *out);
// Command Queue
error *create_command_queue(clobj_t *queue, clobj_t context, clobj_t device,
                            cl_command_queue_properties properties);
error *command_queue__finish(clobj_t queue);
error *command_queue__flush(clobj_t queue);
// Buffer
error *create_buffer(clobj_t *buffer, clobj_t context, cl_mem_flags flags,
                     size_t size, void *hostbuf);
error *buffer__get_sub_region(clobj_t *_sub_buf, clobj_t _buf, size_t orig,
                              size_t size, cl_mem_flags flags);
error *buffer__getitem(clobj_t *_ret, clobj_t _buf, ssize_t start, ssize_t end);
// Memory Object
error *memory_object__release(clobj_t obj);
// Memory Map
error *memory_map__release(clobj_t _map, clobj_t _queue,
                           const clobj_t *_wait_for, uint32_t num_wait_for,
                           clobj_t *evt);
void *memory_map__data(clobj_t _map);
// Program
error *create_program_with_source(clobj_t *program, clobj_t context,
                                  const char *src);
error *create_program_with_binary(clobj_t *program, clobj_t context,
                                  cl_uint num_devices, const clobj_t *devices,
                                  const unsigned char **binaries,
                                  size_t *binary_sizes);
error *program__build(clobj_t program, const char *options,
                      cl_uint num_devices, const clobj_t *devices);
error *program__kind(clobj_t program, int *kind);
error *program__get_build_info(clobj_t program, clobj_t device,
                               cl_program_build_info param, generic_info *out);
// Sampler
error *create_sampler(clobj_t *sampler, clobj_t context, int norm_coords,
                      cl_addressing_mode am, cl_filter_mode fm);
// Kernel
error *create_kernel(clobj_t *kernel, clobj_t program, const char *name);
error *kernel__set_arg_null(clobj_t kernel, cl_uint arg_index);
error *kernel__set_arg_mem(clobj_t kernel, cl_uint arg_index, clobj_t mem);
error *kernel__set_arg_sampler(clobj_t kernel, cl_uint arg_index,
                               clobj_t sampler);
error *kernel__set_arg_buf(clobj_t kernel, cl_uint arg_index,
                           const void *buffer, size_t size);
error *kernel__get_work_group_info(clobj_t kernel,
                                   cl_kernel_work_group_info param,
                                   clobj_t device, generic_info *out);
// Image
error *create_image_2d(clobj_t *image, clobj_t context, cl_mem_flags flags,
                       cl_image_format *fmt, size_t width, size_t height,
                       size_t pitch, void *buffer);
error *create_image_3d(clobj_t *image, clobj_t context, cl_mem_flags flags,
                       cl_image_format *fmt, size_t width, size_t height,
                       size_t depth, size_t pitch_x, size_t pitch_y,
                       void *buffer);
error *create_image_from_desc(clobj_t *img, clobj_t _ctx, cl_mem_flags flags,
                              cl_image_format *fmt, cl_image_desc *desc,
                              void *buffer);
error *image__get_image_info(clobj_t img, cl_image_info param,
                             generic_info *out);
type_t image__get_fill_type(clobj_t img);
// Event
error *event__get_profiling_info(clobj_t event, cl_profiling_info param,
                                 generic_info *out);
error *event__wait(clobj_t event);
error *event__set_callback(clobj_t _evt, cl_int type, void *pyobj);
error *wait_for_events(const clobj_t *_wait_for, uint32_t num_wait_for);
// Nanny Event
void *nanny_event__get_ward(clobj_t evt);
// enqueue_*
error *enqueue_nd_range_kernel(clobj_t *event, clobj_t queue,
                               clobj_t kernel, cl_uint work_dim,
                               const size_t *global_work_offset,
                               const size_t *global_work_size,
                               const size_t *local_work_size,
                               const clobj_t *wait_for, uint32_t num_wait_for);
error *enqueue_task(clobj_t *_evt, clobj_t _queue, clobj_t _knl,
                    const clobj_t *_wait_for, uint32_t num_wait_for);

error *enqueue_marker_with_wait_list(clobj_t *event, clobj_t queue,
                                     const clobj_t *wait_for,
                                     uint32_t num_wait_for);
error *enqueue_barrier_with_wait_list(clobj_t *event, clobj_t queue,
                                      const clobj_t *wait_for,
                                      uint32_t num_wait_for);
error *enqueue_wait_for_events(clobj_t _queue, const clobj_t *_wait_for,
                               uint32_t num_wait_for);
error *enqueue_marker(clobj_t *event, clobj_t queue);
error *enqueue_barrier(clobj_t queue);
// enqueue_*_buffer*
error *enqueue_read_buffer(clobj_t *event, clobj_t queue, clobj_t mem,
                           void *buffer, size_t size, size_t device_offset,
                           const clobj_t *wait_for, uint32_t num_wait_for,
                           int is_blocking, void *pyobj);
error *enqueue_copy_buffer(clobj_t *event, clobj_t queue, clobj_t src,
                           clobj_t dst, ptrdiff_t byte_count,
                           size_t src_offset, size_t dst_offset,
                           const clobj_t *wait_for, uint32_t num_wait_for);
error *enqueue_write_buffer(clobj_t *event, clobj_t queue, clobj_t mem,
                            const void *buffer, size_t size,
                            size_t device_offset, const clobj_t *wait_for,
                            uint32_t num_wait_for, int is_blocking,
                            void *pyobj);
error *enqueue_map_buffer(clobj_t *_evt, clobj_t *mpa, clobj_t _queue,
                          clobj_t _mem, cl_map_flags flags, size_t offset,
                          size_t size, const clobj_t *_wait_for,
                          uint32_t num_wait_for, int block);
error *enqueue_fill_buffer(clobj_t *_evt, clobj_t _queue, clobj_t _mem,
                           void *pattern, size_t psize, size_t offset,
                           size_t size, const clobj_t *_wait_for,
                           uint32_t num_wait_for);
error *enqueue_read_buffer_rect(clobj_t *evt, clobj_t _queue, clobj_t _mem,
                                void *buf, const size_t *_buf_orig,
                                size_t buf_orig_l, const size_t *_host_orig,
                                size_t host_orig_l, const size_t *_reg,
                                size_t reg_l, const size_t *_buf_pitches,
                                size_t buf_pitches_l,
                                const size_t *_host_pitches,
                                size_t host_pitches_l, const clobj_t *_wait_for,
                                uint32_t num_wait_for, int block, void *pyobj);
error *enqueue_write_buffer_rect(clobj_t *evt, clobj_t _queue, clobj_t _mem,
                                 void *buf, const size_t *_buf_orig,
                                 size_t buf_orig_l, const size_t *_host_orig,
                                 size_t host_orig_l, const size_t *_reg,
                                 size_t reg_l, const size_t *_buf_pitches,
                                 size_t buf_pitches_l,
                                 const size_t *_host_pitches,
                                 size_t host_pitches_l,
                                 const clobj_t *_wait_for,
                                 uint32_t num_wait_for, int block, void *pyobj);
error *enqueue_copy_buffer_rect(clobj_t *evt, clobj_t _queue, clobj_t _src,
                                clobj_t _dst, const size_t *_src_orig,
                                size_t src_orig_l, const size_t *_dst_orig,
                                size_t dst_orig_l, const size_t *_reg,
                                size_t reg_l, const size_t *_src_pitches,
                                size_t src_pitches_l,
                                const size_t *_dst_pitches,
                                size_t dst_pitches_l, const clobj_t *_wait_for,
                                uint32_t num_wait_for);
// enqueue_*_image*
error *enqueue_read_image(clobj_t *event, clobj_t queue, clobj_t mem,
                          const size_t *origin, size_t origin_l,
                          const size_t *region, size_t region_l,
                          void *buffer, size_t row_pitch, size_t slice_pitch,
                          const clobj_t *wait_for, uint32_t num_wait_for,
                          int is_blocking, void *pyobj);
error *enqueue_copy_image(clobj_t *_evt, clobj_t _queue, clobj_t _src,
                          clobj_t _dst, const size_t *_src_origin,
                          size_t src_origin_l, const size_t *_dst_origin,
                          size_t dst_origin_l, const size_t *_region,
                          size_t region_l, const clobj_t *_wait_for,
                          uint32_t num_wait_for);
error *enqueue_write_image(clobj_t *_evt, clobj_t _queue, clobj_t _mem,
                           const size_t *origin, size_t origin_l,
                           const size_t *region, size_t region_l,
                           const void *buffer, size_t row_pitch,
                           size_t slice_pitch, const clobj_t *_wait_for,
                           uint32_t num_wait_for, int is_blocking,
                           void *pyobj);
error *enqueue_map_image(clobj_t *_evt, clobj_t *map, clobj_t _queue,
                         clobj_t _mem, cl_map_flags flags,
                         const size_t *_origin, size_t origin_l,
                         const size_t *_region, size_t region_l,
                         size_t *row_pitch, size_t *slice_pitch,
                         const clobj_t *_wait_for, uint32_t num_wait_for,
                         int block);
error *enqueue_fill_image(clobj_t *evt, clobj_t _queue, clobj_t mem,
                          const void *color, const size_t *_origin,
                          size_t origin_l, const size_t *_region,
                          size_t region_l, const clobj_t *_wait_for,
                          uint32_t num_wait_for);
// CL Object
intptr_t clobj__int_ptr(clobj_t obj);
error *clobj__get_info(clobj_t obj, cl_uint param, generic_info *out);
void clobj__delete(clobj_t obj);
error *clobj__from_int_ptr(clobj_t *out, intptr_t ptr, class_t);
