from urllib.request import urlopen


_conf_url = \
        "https://raw.githubusercontent.com/inducer/sphinxconfig/main/sphinxconfig.py"
with urlopen(_conf_url) as _inf:
    exec(compile(_inf.read(), _conf_url, "exec"), globals())

exclude_patterns = ["subst.rst"]

copyright = "2009-21, Andreas Kloeckner"

ver_dic = {}
with open("../pyopencl/version.py") as ver_file:
    ver_src = ver_file.read()
exec(compile(ver_src, "../pyopencl/version.py", "exec"), ver_dic)
version = ".".join(str(x) for x in ver_dic["VERSION"])
# The full version, including alpha/beta/rc tags.
release = ver_dic["VERSION_TEXT"]


nitpick_ignore = [
    ("py:class", r"numpy._typing._dtype_like._SupportsDType"),
    ("py:class", r"numpy._typing._dtype_like._DTypeDict"),
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "mako": ("https://docs.makotemplates.org/en/latest", None),
    "pytools": ("https://documen.tician.de/pytools", None),
}
