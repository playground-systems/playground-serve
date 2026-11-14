#!/usr/bin/env bash
# Release check: fails if the tree holds anything that must not be published: API keys and tokens, home-directory
# paths, e-mail or IP addresses, Runpod resource ids or owner ids, account balances, key or state files, or names on a
# private denylist. The denylist is stored as SHA-256 hashes of lower-case words (with - _ . / removed), so this script
# does not publish the words it looks for. Usage: scripts/check_release.sh [DIR] (default: the repository root).
set -euo pipefail
ROOT="${1:-$(cd "$(dirname "$0")/.." && pwd)}"
exec python3 - "$ROOT" <<'PY'
import hashlib, os, re, sys

root = sys.argv[1]
SKIP_DIRS = {".git", ".venv", "venv", "dist", "build", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
DENY = {  # sha256 of private words: the owner's names, private key files, other projects
    "63b7cc6dd0c192e8f73170629701879816156027e44bbee145a7a8456a9a3517",
    "2ea6c9756285d0fa2f075fe40e1caeaaeee9a174c61015c03bfcfbd872fd01ef",
    "00b5f3596ed06e78b73948d04688a012dddff8f42efb61325608d2d1bbf9bacd",
    "dfe5880b9a76898a6fe029ba807b43fd98778c1a5d22abf5a9a5a9fffe721e8e",
    "a373d6516418e51aba694dc370a60f72dcd75ec2f88a108a31d331f069ff9c38",
    "9eda4ecf0c6e80ec321180432241730e0bc87dede33710db09d0419e01449806",
    "6f6327a1fbfda7bec90a8ff3eb7a8170cb3e17a2e1cbbe0dd249fd69d336017e",
    "c36461be47eaa5034c49fd979022191870c59d12f46b01db38c5443ffa0925b7",
    "9d03dd4b3e6967965c3e1d9c1a851e1aed4e804fed9880379ed1ab33864226d8",
}
DENY_LENGTHS = (7, 8, 10, 11, 13, 14)
PATTERNS = {
    "Runpod API key": r"rpa_[A-Za-z0-9]{16,}",
    "Runpod S3 secret or user id": r"\b(?:rps_[A-Za-z0-9]{16,}|user_[A-Za-z0-9]{20,})",
    "Hugging Face token": r"\bhf_[A-Za-z0-9]{20,}",
    "secret-shaped token": r"\b(?:sk-[A-Za-z0-9_-]{20,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_\w+|AKIA[0-9A-Z]{16}|xox[abprs]-[\w-]+)",
    "private key block": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "hard-coded secret": r"(?i)(?:api[_-]?key|token|secret|password)\s*[:=]\s*[\"'][A-Za-z0-9_\-]{20,}[\"']",
    "home directory path": r"/Users/[A-Za-z]|/home/[a-z]|[A-Za-z]:\\\\Users\\\\",
    "e-mail address": r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+\.[A-Za-z]{2,}",
    "IP address": r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b",
    "owner id": r"\bo(?!00000000\b|11111111\b|22222222\b|1234abcd\b|9999ffff\b)[0-9a-f]{8}\b",  # test fakes allowed
    "Runpod resource id": r"\b(?=[a-z0-9]{10,14}\b)(?=[a-z]*[0-9])(?=[0-9]*[a-z])[a-z0-9]{10,14}\b",
    "account balance": r"(?i)\bbalance\b[^\n.]{0,20}\d+\.\d\d",
    "old project name": r"(?i)robo[-]serve|\brs[-]pi05",
}
# Tokens the resource-id pattern matches that are not ids (package tags, model names, CUDA builds).
NOT_IDS = {"manylinux2014", "manylinux2010", "ubuntu2404"}
# The pre-rename wire key (robo, underscore, serve) is still accepted by the client, in these files only.
OLD_WIRE_KEY_FILES = ("playground_serve/client/client.py", "playground_serve/client/mock_server.py", "docs/WIRE.md",
                      "tests/test_connect.py")  # matched as path endings, so a built wheel or sdist checks too
FORBIDDEN_NAMES = re.compile(r"(?i)(^|/)(\.claude|\.env|state\.json|[^/]*\.key|key)$")

compiled = {name: re.compile(p) for name, p in PATTERNS.items()}
hits = []


def denylisted(text):
    norm = re.sub(r"[-_./]", "", text.lower())
    return any(hashlib.sha256(norm[i:i + n].encode()).hexdigest() in DENY
               for n in DENY_LENGTHS for i in range(len(norm) - n + 1))


for dirpath, dirnames, filenames in os.walk(root):
    dirnames[:] = sorted(d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info"))
    for name in sorted(filenames + [d for d in dirnames if d == ".claude"]):
        path = os.path.join(dirpath, name)
        rel = os.path.relpath(path, root)
        if FORBIDDEN_NAMES.search(rel):
            hits.append(f"{rel}: file or folder that must not be published")
        if denylisted(rel):
            hits.append(f"{rel}: private name in the path")
        if not os.path.isfile(path):
            continue
        data = open(path, "rb").read()
        binary = b"\0" in data[:8192]
        # Binary files: only their printable runs (like `strings`), so random bytes do not match.
        text = (" ".join(m.decode() for m in re.findall(rb"[\x20-\x7e]{6,}", data)) if binary
                else data.decode("utf-8", "replace"))
        for label, rx in compiled.items():
            for m in rx.finditer(text):
                if label == "Runpod resource id" and (binary or m.group() in NOT_IDS
                                                      or re.fullmatch(r"\d+e\d+", m.group())):  # 2.3819763e38
                    continue
                if label == "IP address" and rel == "uv.lock":
                    continue  # four-part version numbers
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{rel}:{line}: {label}: {m.group()[:40]}")
        for m in re.finditer(r"robo[_]serve", text):
            if not rel.endswith(OLD_WIRE_KEY_FILES):
                hits.append(f"{rel}:{text.count(chr(10), 0, m.start()) + 1}: old wire key outside the allowed files")
        for number, line in enumerate(text.splitlines(), 1):
            if denylisted(line):
                hits.append(f"{rel}:{number}: private name (denylist)")

if hits:
    print("Release check FAILED:\n  " + "\n  ".join(hits))
    sys.exit(1)
print(f"Release check passed: {root}")
PY
