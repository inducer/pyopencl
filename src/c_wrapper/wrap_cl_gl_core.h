// Interface between C and Python for GL related functions

error* create_from_gl_texture(clobj_t *ptr, clobj_t _ctx, cl_mem_flags flags,
                       GLenum texture_target, GLint miplevel,
                       GLuint texture);
error *create_from_gl_buffer(clobj_t *ptr, clobj_t context,
                             cl_mem_flags flags, GLuint bufobj);
error *create_from_gl_renderbuffer(clobj_t *ptr, clobj_t context,
                                   cl_mem_flags flags, GLuint bufobj);
error *enqueue_acquire_gl_objects(
    clobj_t *event, clobj_t queue, const clobj_t *mem_objects,
    uint32_t num_mem_objects, const clobj_t *wait_for, uint32_t num_wait_for);
error *enqueue_release_gl_objects(
    clobj_t *event, clobj_t queue, const clobj_t *mem_objects,
    uint32_t num_mem_objects, const clobj_t *wait_for, uint32_t num_wait_for);
cl_context_properties get_apple_cgl_share_group();
error *get_gl_object_info(clobj_t mem, cl_gl_object_type *otype,
                          GLuint *gl_name);
