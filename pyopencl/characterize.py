def has_double_support(dev):
    for ext in dev.extensions.split(" "):
        if ext == "cl_khr_fp64":
            return True
    return False
