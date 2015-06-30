from __future__ import absolute_import
from __future__ import print_function
import pyopencl as cl
from optparse import OptionParser

parser = OptionParser()
parser.add_option("-s", "--short", action="store_true",
                  help="don't print all device properties")

(options, args) = parser.parse_args()


def print_info(obj, info_cls):
    for info_name in sorted(dir(info_cls)):
        if not info_name.startswith("_") and info_name != "to_string":
            info = getattr(info_cls, info_name)
            try:
                info_value = obj.get_info(info)
            except:
                info_value = "<error>"

            if (info_cls == cl.device_info and info_name == "PARTITION_TYPES_EXT"
                    and isinstance(info_value, list)):
                print("%s: %s" % (info_name, [
                    cl.device_partition_property_ext.to_string(v,
                        "<unknown device partition property %d>")
                    for v in info_value]))
            else:
                try:
                    print("%s: %s" % (info_name, info_value))
                except:
                    print("%s: <error>" % info_name)

for platform in cl.get_platforms():
    print(75*"=")
    print(platform)
    print(75*"=")
    if not options.short:
        print_info(platform, cl.platform_info)

    for device in platform.get_devices():
        if not options.short:
            print(75*"-")
        print(device)
        if not options.short:
            print(75*"-")
            print_info(device, cl.device_info)
            ctx = cl.Context([device])
            for mf in [
                    cl.mem_flags.READ_ONLY,
                    #cl.mem_flags.READ_WRITE,
                    #cl.mem_flags.WRITE_ONLY
                    ]:
                for itype in [
                        cl.mem_object_type.IMAGE2D,
                        cl.mem_object_type.IMAGE3D
                        ]:
                    try:
                        formats = cl.get_supported_image_formats(ctx, mf, itype)
                    except:
                        formats = "<error>"
                    else:
                        def str_chd_type(chdtype):
                            result = cl.channel_type.to_string(chdtype,
                                    "<unknown channel data type %d>")

                            result = result.replace("_INT", "")
                            result = result.replace("UNSIGNED", "U")
                            result = result.replace("SIGNED", "S")
                            result = result.replace("NORM", "N")
                            result = result.replace("FLOAT", "F")
                            return result

                        formats = ", ".join(
                                "%s-%s" % (
                                    cl.channel_order.to_string(iform.channel_order,
                                        "<unknown channel order 0x%x>"),
                                    str_chd_type(iform.channel_data_type))
                                for iform in formats)

                    print("%s %s FORMATS: %s\n" % (
                            cl.mem_object_type.to_string(itype),
                            cl.mem_flags.to_string(mf),
                            formats))
            del ctx
