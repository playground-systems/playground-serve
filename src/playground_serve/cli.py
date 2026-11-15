"""pgs (Playground Serve) command line: up, demo, down, status, delete, models."""

from __future__ import annotations

import argparse
import functools
import getpass
import sys

from . import __version__, state
from .launcher import IDLE_MIN, KEY_PAGE, KEY_PERMISSIONS, Launcher, LauncherError, advice
from .resolve import WS_URL, resolve_key
from .runpod_api import ApiError, RunpodApi

say = functools.partial(print, flush=True)


def read_key(key_file: str | None) -> str:
    """--key-file, then RUNPOD_API_KEY, then the saved key file. With none, on a terminal: ask for the key once, check
    it with one read, and save it (0600) where later commands and connect() find it."""
    try:
        key = resolve_key(None, key_file)
    except ValueError as e:
        raise LauncherError(str(e)) from None
    if key:
        return key
    how = f"Create one at {KEY_PAGE} (API Keys, Create API Key). {KEY_PERMISSIONS}"
    if not sys.stdin.isatty():
        raise LauncherError(f"no Runpod API key. {how} Then set RUNPOD_API_KEY, or run pgs once in a terminal to "
                            f"paste the key and save it to {state.key_path()}.")
    say(f"No Runpod API key found. {how}\nPaste it below; pgs checks it with one read and saves it to "
        f"{state.key_path()}, readable only by you.")
    try:
        key = getpass.getpass("Runpod API key (input hidden): ").strip()
    except EOFError:  # Ctrl-D or closed input
        key = ""
    if not key:
        raise LauncherError("no key entered; nothing saved.")
    try:
        RunpodApi(key).list_endpoints()
    except ApiError as e:
        if not e.status:
            raise LauncherError(f"could not reach Runpod to check the key ({e}); nothing saved.") from None
        hint = f"Check that you pasted the whole key, or create one at {KEY_PAGE}." if e.status == 401 else advice(e)
        raise LauncherError(f"Runpod refused this key ({e}); nothing saved. {hint}") from None
    say(f"Key accepted and saved to {state.save_key(key)}.")
    return key


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pgs", description="Serve an openpi robot policy on a Runpod serverless GPU.",
                                epilog=f"State: ${state.HOME_ENV}/state.json (one per project if you like), default "
                                       "~/.config/playground-serve/state.json. Its owner id is in the name of every "
                                       "resource it creates; up, down and delete act only on those.")
    p.add_argument("--version", action="version", version=f"playground-serve {__version__}")
    p.add_argument("--key-file", help="file that holds the Runpod API key "
                   "(default: $RUNPOD_API_KEY, then ~/.config/playground-serve/key)")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--key-file", default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("up", parents=[common], help="start (creating if needed) the GPU server for a model")
    s.add_argument("model", help="model name, for example pi05_libero")
    s.add_argument("--warm", action="store_true", help="keep 1 worker running (and billing) until `pgs down`; by "
                   "default none is kept and a worker stops by itself --idle minutes after its last client")
    s.add_argument("--idle", type=int, default=IDLE_MIN, metavar="MIN",
                   help=f"minutes a worker stays after its last request or client (default {IDLE_MIN})")
    s.add_argument("--region", metavar="DC", help="data center for a new endpoint and volume, e.g. US-NC-1 (default: "
                   "the nearest with stock: by the round trip a short-lived probe per data center measured in the last "
                   "day, else by this machine's time zone; the probe runs while the worker starts)")
    s.add_argument("--gpu", metavar="GPU", help="GPU for the endpoint instead of the model's default pools, by name or "
                   "Runpod pool, e.g. H200 or HOPPER_141 (`pgs up` prints its price before it creates anything)")
    s.add_argument("--stock-path", action="store_true", help="serve openpi's own policy instead of the fast path "
                   "(realtime-vla graph, the default for pi05_libero on an RTX 4090; same weights, checked at boot)")
    s.add_argument("--direct", action="store_true", help="also serve on a TCP port of the worker's own public address, "
                   "so requests skip Runpod's gateway (about 14 ms less per request from the US east coast to "
                   "US-CA-2); TLS with the worker's own certificate, no Cloudflare in front; still serverless")
    g = s.add_mutually_exclusive_group()
    g.add_argument("--adopt", nargs="?", const=True, metavar="ID", help="take over this model's endpoint that another "
                   "owner (or a `pgs` from before owner ids) created")
    g.add_argument("--new", action="store_true", help="create your own even if another owner has one for this model")
    s = sub.add_parser("demo", parents=[common], help="send a dummy observation to a running model and print the "
                       "actions' shape and the round trip (LIBERO models; starts a worker if none runs)")
    s.add_argument("model", nargs="?", default="pi05_libero", help="model name or wss:// URL (default pi05_libero)")
    s = sub.add_parser("down", parents=[common], help="keep no worker: GPU billing stops once idle, the volume stays")
    s.add_argument("model", nargs="?", help="model name (default: all of yours)")
    sub.add_parser("status", parents=[common], help="endpoints (yours and not yours), worker state, price now; "
                   "read only")
    s = sub.add_parser("delete", parents=[common], help="delete the endpoint (and with --volume the volume)")
    s.add_argument("model")
    s.add_argument("--volume", action="store_true", help="also delete the network volume")
    s.add_argument("--force", action="store_true", help="also delete another owner's endpoint or volume of this model")
    sub.add_parser("models", parents=[common], help="every model with its GPU, hourly price, whether it ran on a GPU, "
                   "and its license note; read only, works without a key (then with list prices)")
    return p


def live_prices(key_file: str | None) -> dict | None:
    """Pool -> USD/hr while a worker runs (the dearest GPU type of the pool, as `up` quotes it), or None without a
    key or when the catalog read fails."""
    try:
        key = resolve_key(None, key_file)
        gpus = RunpodApi(key).serverless_gpus() if key else []
    except (ValueError, OSError, ApiError):
        return None
    prices: dict = {}
    for g in gpus:
        price = (g.get("price") or {}).get("serverless")
        if g.get("pool") and price:
            prices[g["pool"]] = max(prices.get(g["pool"], 0.0), price)
    return prices or None


def main(argv: list[str] | None = None) -> int:
    p = parser()
    args = p.parse_args(argv)
    if getattr(args, "idle", 1) < 1:
        p.error("--idle must be at least 1 minute")
    if args.command == "models":
        from . import models

        print(models.table(live_prices(args.key_file)))
        return 0
    key = ""
    try:
        key = read_key(args.key_file)
        if args.command == "demo":
            demo(args.model, key)
            return 0
        launcher = Launcher(RunpodApi(key), state.default_path(), out=say)
        if args.command == "up":
            from . import models

            try:
                gpu = models.pool(args.gpu) if args.gpu else None
            except KeyError as e:
                raise LauncherError(str(e).strip("'\"")) from None
            launcher.up(args.model, warm=args.warm, idle_min=args.idle, adopt=args.adopt, new=args.new,
                        stock_path=args.stock_path, dc=args.region, gpu=gpu, direct=args.direct)
        elif args.command == "down":
            launcher.down(args.model)
        elif args.command == "status":
            launcher.status()
        else:
            launcher.delete(args.model, args.volume, force=args.force)
        return 0
    except KeyboardInterrupt:
        return 130
    except (LauncherError, ApiError, OSError) as e:
        text = str(e) + (f". {advice(e)}" if isinstance(e, ApiError) and advice(e) else "")
        text = text.replace(key, "<key>") if key else text  # belt and braces: the key is never in a message
        print(f"pgs: {text}", file=sys.stderr)
        return 1


def demo(target: str, key: str) -> None:
    """pgs demo: example_obs (LIBERO) through the model's endpoint; every failure becomes one sentence."""
    from . import demo as demo_mod, models

    if not WS_URL.match(target):
        try:
            robot = models.get(target).robot
        except (KeyError, ValueError) as e:
            raise LauncherError(str(e).strip("'\"")) from None
        if robot != "libero":
            raise LauncherError(f"pgs demo has an example observation for LIBERO models (pi05_libero) only; for "
                                f"{target}, call connect() with your robot's observation (README).")
    try:
        demo_mod.run(target, key, out=say)
        say("Done. With no client connected the worker stops by itself after its idle time (5 min unless `pgs up "
            "--idle` set another); `pgs down` stops it within about a minute.")
    except Exception as e:  # noqa: BLE001  (a sentence for a first-day user, not a traceback)
        status = getattr(getattr(e, "response", None), "status_code", 0)  # a refused websocket upgrade
        hint = advice(ApiError(status, "GET", "", "")) if status in (401, 403) else ""
        raise LauncherError(f"demo failed: {type(e).__name__}: {e}" + (f". {hint}" if hint else "")) from None


if __name__ == "__main__":
    sys.exit(main())
