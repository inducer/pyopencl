// Everything in here should have a 'pyopencl_' prefix to avoid clashing with
// other libraries imported via CFFI.

error *_create_from_gl_buffer(void **ptr, void *ptr_context, cl_mem_flags flags, GLuint bufobj);
error *_create_from_gl_renderbuffer(void **ptr, void *ptr_context, cl_mem_flags flags, GLuint bufobj);
error *_enqueue_acquire_gl_objects(void **ptr_event, void *ptr_command_queue, void **ptr_mem_objects, uint32_t num_mem_objects, void **wait_for, uint32_t num_wait_for);
error *_enqueue_release_gl_objects(void **ptr_event, void *ptr_command_queue, void **ptr_mem_objects, uint32_t num_mem_objects, void **wait_for, uint32_t num_wait_for);
