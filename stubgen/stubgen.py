# Note: This isn't used on an ongoing basis. It was used to bootstrap
# the stubs, and it can be used to get stubs fow newly-wrapped
# functionality. But the actual stubs in _cl.pyi are handcrafted.
# This script outputs _cl_gen.pyi to make this clear.
from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from nanobind.stubgen import StubGen as StubGenBase
from typing_extensions import override


if TYPE_CHECKING:
    from collections.abc import Callable


class StubGen(StubGenBase):
    # can be removed once https://github.com/wjakob/nanobind/pull/1055 is merged
    @override
    def put_function(self,
                fn: Callable[..., Any],
                name: str | None = None,
                parent: object | None = None
            ):
        fn_module = getattr(fn, "__module__", None)

        if (name and fn_module
                and fn_module != self.module.__name__
                and parent is not None):
            self.import_object(fn_module, name=None)
            rhs = f"{fn_module}.{fn.__qualname__}"
            if type(fn) is staticmethod:
                rhs = f"staticmethod({rhs})"
            self.write_ln(f"{name} = {rhs}\n")

            return

        super().put_function(fn, name, parent)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--module", default="pyopencl._cl")
    parser.add_argument("--python-path", nargs="+")
    parser.add_argument("-o", "--output-dir", default="../pyopencl")
    args = parser.parse_args()
    output_path = Path(cast("str", args.output_dir))

    sys.path.extend(cast("list[str]", args.python_path or []))

    mod = importlib.import_module(cast("str", args.module))
    sg = StubGen(
        module=mod,
        quiet=True,
        recursive=False,
        include_docstrings=False,
        include_private=True,
    )
    sg.put(mod)
    with open(output_path / "_cl_gen.pyi", "w") as outf:
        outf.write(sg.get())


if __name__ == "__main__":
    main()
