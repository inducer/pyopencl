from __future__ import annotations

import re
from importlib import metadata


VERSION_TEXT = metadata.version("pyopencl")
_match = re.match(r"^([0-9.]+)([a-z0-9]*?)$", VERSION_TEXT)
assert _match is not None
VERSION_STATUS = _match.group(2)
VERSION = tuple(int(nr) for nr in _match.group(1).split("."))
