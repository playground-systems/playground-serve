"""Entry point of the `pgs` command. The launcher needs Python 3.10 or later; the robot client also runs on 3.8."""

import sys


def main():
    if sys.version_info < (3, 10):
        sys.exit(f"pgs needs Python 3.10 or later; this is Python {sys.version_info[0]}.{sys.version_info[1]}, where "
                 "only the client (`from playground_serve import connect`) is supported.")
    from .cli import main as cli_main

    return cli_main()
