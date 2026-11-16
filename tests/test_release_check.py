"""scripts/check_release.sh: in a git work tree it checks the tracked files (a worktree's .git file, which names a local
path, is not one of them); elsewhere every file under the folder."""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

SCRIPT = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "check_release.sh"
LEAK = "see /" + "Users/someone/x\n"  # a home-directory path (split so this file does not hold one)

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")


def check(folder) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", str(SCRIPT), str(folder)], capture_output=True, text=True)


def git(folder, *args):
    subprocess.run(["git", "-C", str(folder), *args], check=True, capture_output=True)


def test_tracked_files_only_in_a_work_tree(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    (repo / "a.txt").write_text("fine\n")
    (repo / "local.txt").write_text(LEAK)  # not tracked: not published
    git(repo, "add", "a.txt")
    assert check(repo).returncode == 0
    git(repo, "add", "local.txt")
    result = check(repo)
    assert result.returncode == 1 and "local.txt:1: home directory path" in result.stdout


def test_a_worktree_passes_and_its_tracked_files_are_checked(tmp_path):
    repo, tree = tmp_path / "home" / "user" / "repo", tmp_path / "tree"  # a path the check flags
    repo.mkdir(parents=True)
    git(repo, "init", "-q")
    (repo / "a.txt").write_text("fine\n")
    git(repo, "add", "a.txt")
    git(repo, "-c", "user.name=t", "-c", "user.email=t", "commit", "-q", "-m", "a")
    git(repo, "worktree", "add", "-q", str(tree))
    assert (tree / ".git").is_file() and check(tree).returncode == 0  # its .git file names a local path
    (tree / ".claude").mkdir()
    (tree / ".claude" / "settings.json").write_text("{}\n")
    git(tree, "add", ".claude/settings.json")
    result = check(tree)
    assert result.returncode == 1 and ".claude: file or folder that must not be published" in result.stdout


def test_every_file_outside_a_work_tree(tmp_path):
    folder = tmp_path / "unpacked"  # an unpacked wheel or sdist
    folder.mkdir()
    (folder / "a.txt").write_text(LEAK)
    result = check(folder)
    assert result.returncode == 1 and "a.txt:1: home directory path" in result.stdout
