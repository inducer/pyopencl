from __future__ import division
import numpy
import numpy.linalg as la




def have_cl():
    try:
        import pyopencl
        return True
    except:
        return False


if have_cl():
    import pyopencl as cl




class TestCL:
    disabled = not have_cl()

    def test_get_info(self, platform, device):
        had_failures = [False]

        QUIRKS = [
                ("NVIDIA", [
                    (cl.Device, cl.device_info.PLATFORM),
                    ]),
                ]

        def find_quirk(quirk_list, cl_obj, info):
            for quirk_plat_name, quirks in quirk_list:
                if quirk_plat_name in platform.name:
                    for quirk_cls, quirk_info in quirks:
                        if (isinstance(cl_obj, quirk_cls)
                                and quirk_info == info):
                            return True

            return False

        def do_test(cl_obj, info_cls, func=None, try_attr_form=True):
            if func is None:
                def func(info):
                    cl_obj.get_info(info)

            for info_name in dir(info_cls):
                if not info_name.startswith("_") and info_name != "to_string":
                    info = getattr(info_cls, info_name)

                    try:
                        func(info)
                    except:
                        print "failed get_info", type(cl_obj), info_name

                        if find_quirk(QUIRKS, cl_obj, info):
                            print "(known quirk for %s)" % platform.name
                        else:
                            had_failures[0] = True
                            raise

                    if try_attr_form:
                        try:
                            getattr(cl_obj, info_name.lower())
                        except:
                            print "failed attr-based get_info", type(cl_obj), info_name

                            if find_quirk(QUIRKS, cl_obj, info):
                                print "(known quirk for %s)" % platform.name
                            else:
                                had_failures[0] = True
                                raise

        do_test(platform, cl.platform_info)

        do_test(device, cl.device_info)

        ctx = cl.Context([device])
        do_test(ctx, cl.context_info)

        props = 0
        if (device.queue_properties
                & cl.command_queue_properties.PROFILING_ENABLE):
            profiling = True
            props = cl.command_queue_properties.PROFILING_ENABLE
        queue = cl.CommandQueue(ctx,
                properties=props)
        do_test(queue, cl.command_queue_info)

        prg = cl.Program(ctx, """
            __kernel void sum(__global float *a)
            { a[get_global_id(0)] *= 2; }
            """).build()
        do_test(prg, cl.program_info)
        do_test(prg, cl.program_build_info,
                lambda info: prg.get_build_info(device, info),
                try_attr_form=False)

        cl.unload_compiler() # just for the heck of it

        mf = cl.mem_flags
        n = 2000
        a_buf = cl.Buffer(ctx, 0, n*4)

        do_test(a_buf, cl.mem_info)

        kernel = prg.sum
        do_test(kernel, cl.kernel_info)

        evt = kernel(queue, (n,), a_buf)
        do_test(evt, cl.event_info)

        if profiling:
            evt.wait()
            do_test(evt, cl.profiling_info,
                    lambda info: evt.get_profiling_info(info),
                    try_attr_form=False)

        if device.image_support:
            smp = cl.Sampler(ctx, True,
                    cl.addressing_mode.CLAMP,
                    cl.filter_mode.NEAREST)
            do_test(smp, cl.sampler_info)

            img_format = cl.get_supported_image_formats(
                    ctx, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D)[0]

            img = cl.Image(ctx, cl.mem_flags.READ_ONLY, img_format, (128, 256))
            assert img.shape == (128, 256)

            img.depth
            img.image.depth
            do_test(img, cl.image_info,
                    lambda info: img.get_image_info(info))

        if had_failures[0]:
            raise RuntimeError("get_info testing had errors")

    def test_invalid_kernel_names_cause_failures(self):
        for platform in cl.get_platforms():
            for device in platform.get_devices():
                ctx = cl.Context([device])
                prg = cl.Program(ctx, """
                    __kernel void sum(__global float *a)
                    { a[get_global_id(0)] *= 2; }
                    """).build()

                try:
                    prg.sam
                    raise RuntimeError("invalid kernel name did not cause error")
                except AttributeError:
                    pass

    def test_image_format_constructor(self):
        # doesn't need image support to succeed
        iform = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.FLOAT)

        assert iform.channel_order == cl.channel_order.RGBA
        assert iform.channel_data_type == cl.channel_type.FLOAT
        assert not iform.__dict__

    def test_nonempty_supported_image_formats(self, device, context):
        if device.image_support:
            assert len(cl.get_supported_image_formats(
                    context, cl.mem_flags.READ_ONLY, cl.mem_object_type.IMAGE2D)) > 0
        else:
            from py.test import skip
            skip("images not supported on %s" % device.name)





def pytest_generate_tests(metafunc):
    if ("device" in metafunc.funcargnames
            or "context" in metafunc.funcargnames):
        arg_dict = {}

        for platform in cl.get_platforms():
            if "platform" in metafunc.funcargnames:
                arg_dict["platform"] = platform

            for device in platform.get_devices():
                if "device" in metafunc.funcargnames:
                    arg_dict["device"] = device

                if "context" in metafunc.funcargnames:
                    arg_dict["context"] = cl.Context([device])

                metafunc.addcall(funcargs=arg_dict.copy(),
                        id=", ".join("%s=%s" % (arg, value) 
                                for arg, value in arg_dict.iteritems()))

    elif "platform" in metafunc.funcargnames:
        for platform in cl.get_platforms():
            metafunc.addcall(
                    funcargs=dict(platform=platform),
                    id=str(platform))


if __name__ == "__main__":
    # make sure that import failures get reported, instead of skipping the tests.
    import pyopencl

    from py.test.cmdline import main
    import sys
    main([__file__]+sys.argv[1:])
