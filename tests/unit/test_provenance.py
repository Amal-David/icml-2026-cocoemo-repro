import subprocess
from pathlib import Path

from repro.provenance import git_state, require_clean_worktree


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
    assert clean["dirty_file_count"] == 0
    assert "status" not in clean

    tracked.write_text("changed\n")
    dirty = git_state(tmp_path)
    assert dirty["dirty"]
    assert dirty["dirty_file_count"] == 1
    assert "tracked.txt" not in str(dirty)
    assert dirty["dirty_state_sha256"] != clean["dirty_state_sha256"]
    try:
        require_clean_worktree(tmp_path)
    except RuntimeError as exc:
        assert "clean git worktree" in str(exc)
    else:  # pragma: no cover - defensive assertion for the gate
        raise AssertionError("dirty worktree was accepted")
