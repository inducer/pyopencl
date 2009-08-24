VERSION = (0, 90)
VERSION_STATUS = "alpha"
VERSION_TEXT = ".".join(str(x) for x in VERSION) + VERSION_STATUS

from pyopencl._cl import *
