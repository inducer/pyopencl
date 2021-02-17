try:
    import mako.template  # noqa
except ImportError:
    raise ImportError(
            "Some of PyOpenCL's facilities require the Mako templating engine.\n"
            "You or a piece of software you have used has tried to call such a\n"
            "part of PyOpenCL, but there was a problem importing Mako.\n\n"
            "You may install mako now by typing one of:\n"
            "- easy_install Mako\n"
            "- pip install Mako\n"
            "- aptitude install python-mako\n"
            "\nor whatever else is appropriate for your system.")

from mako import *  # noqa
