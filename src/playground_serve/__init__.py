"""Playground Serve: serve an openpi robot policy on a Runpod serverless GPU with one command.

    from playground_serve import connect
    policy = connect("pi05_libero")
    actions = policy.act(obs)
"""

__version__ = "0.1.1"
__all__ = ["ActionQueue", "History", "Policy", "ServerError", "Status", "connect", "example_obs", "status"]


def __getattr__(name: str):
    # Lazy, so that the `pgs` command does not import the client's websockets, numpy and Pillow, and the client does
    # not import the launcher (the client runs on Python 3.8, `pgs` needs 3.10).
    if name in ("connect", "status", "Status"):
        from . import resolve

        return getattr(resolve, name)
    if name in ("ActionQueue", "Policy", "ServerError"):
        from . import client

        return getattr(client, name)
    if name == "History":
        from .client.history import History

        return History
    if name == "example_obs":
        from .demo import example_obs

        return example_obs
    raise AttributeError(f"module 'playground_serve' has no attribute {name!r}")
