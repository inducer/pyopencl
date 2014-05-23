// Everything in here should have a 'pyopencl_' prefix to avoid clashing with
// other libraries imported via CFFI.

error *_create_from_gl_buffer(clobj_t *ptr, clobj_t context,
                              cl_mem_flags flags, GLuint bufobj);
error *_create_from_gl_renderbuffer(clobj_t *ptr, clobj_t context,
                                    cl_mem_flags flags, GLuint bufobj);
error *_enqueue_acquire_gl_objects(
    clobj_t *ptr_event, clobj_t queue, const clobj_t *mem_objects,
    uint32_t num_mem_objects, const clobj_t *wait_for, uint32_t num_wait_for);
error *_enqueue_release_gl_objects(
    clobj_t *ptr_event, clobj_t queue, const clobj_t *mem_objects,
    uint32_t num_mem_objects, const clobj_t *wait_for, uint32_t num_wait_for);
