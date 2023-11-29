"""Showyourwork."""

import pathlib

try:
    from showyourwork.paths import user  # pylint: disable=W0611
except ImportError:
    class user:  # pylint: disable=W0611
        """Paths for the user's project."""

        def __init__(self, base: pathlib.Path) -> None:

            if base.name != "src":
                base = base / "src"

            self.scripts: pathlib.Path = base / "scripts"
            self.data: pathlib.Path = base / "data"
            self.static: pathlib.Path = base / "static"

        def __repr__(self) -> str:
            return f"user(\n\t{self.scripts!r},\n\t{self.data!r},\n\t{self.static!r}\n)"