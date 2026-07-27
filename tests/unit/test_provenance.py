import subprocess
from pathlib import Path

from repro.provenance import git_state


def test_git_state_distinguishes_clean_and_dirty_worktrees(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("baseline\n")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Repro Test",
            "-c",
            "user.email=repro@example.invalid",
            "commit",
            "-q",
            "-m",
            "baseline",
        ],
        cwd=tmp_path,
        check=True,
    )

    clean = git_state(tmp_path)
    assert not clean["dirty"]
    assert clean["status"] == []

    tracked.write_text("changed\n")
    dirty = git_state(tmp_path)
    assert dirty["dirty"]
    assert dirty["status"] == [" M tracked.txt"]
    assert dirty["tracked_diff_sha256"] != clean["tracked_diff_sha256"]
