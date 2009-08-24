import pyopencl as cl

def doc_class(cls):
    print ".. class :: %s" % cls.__name__
    print
    for i in sorted(dir(cls)):
        if not i.startswith("_"):
            print "    .. attribute :: %s" % i
    print


print ".. This is an automatically generated file. DO NOT EDIT"
for cls in [
        cl.platform_info,
        cl.device_type,
        cl.device_info,
        cl.device_fp_config,
        cl.device_mem_cache_type,
        cl.device_local_mem_type,
        cl.command_queue_properties,
        cl.context_info,
        cl.context_properties,
        cl.command_queue_info,
        cl.mem_flags,
        cl.channel_order,
        cl.mem_object_type,
        cl.mem_info,
        cl.image_info,
        cl.addressing_mode,
        cl.filter_mode,
        cl.sampler_info,
        cl.map_flags,
        cl.program_info,
        cl.program_build_info,
        cl.kernel_info,
        cl.kernel_work_group_info,
        cl.event_info,
        cl.command_execution_status,
        cl.profiling_info,
        ]:
    doc_class(cls)
