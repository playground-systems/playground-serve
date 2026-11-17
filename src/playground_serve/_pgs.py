"""Entry point of the `pgs` command (Python 3.11+ for patched Pillow)."""

import sys


def main():
    if sys.version_info < (3, 11):
        sys.exit("playground-serve needs Python 3.11 or later for its security-patched image decoder.")
    from .cli import main as cli_main

    return cli_main()
