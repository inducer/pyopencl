import pyopencl as cl

def print_info(obj, info_cls):
    for info_name in sorted(dir(info_cls)):
        if not info_name.startswith("_") and info_name != "to_string":
            info = getattr(info_cls, info_name)
            try:
                info_value = obj.get_info(info)
            except:
                info_value = "<error>"

            print "%s: %s" % (info_name, info_value)

for platform in cl.get_platforms():
    print 75*"="
    print platform
    print 75*"="
    print_info(platform, cl.platform_info)

    for device in platform.get_devices():
        print 75*"-"
        print device
        print 75*"-"
        print_info(device, cl.device_info)
