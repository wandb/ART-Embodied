from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from art_embodied import storage


def test_preflight_storage_rejects_insufficient_capacity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        storage.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(
            total=200 * 1024**3,
            used=190 * 1024**3,
            free=10 * 1024**3,
        ),
    )

    with pytest.raises(ValueError, match="Insufficient free storage"):
        storage.preflight_storage(
            tmp_path / "run",
            required_root=tmp_path,
            minimum_free_gib=100,
        )


def test_preflight_storage_resolves_output_symlink(
    tmp_path: Path,
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    output_link = tmp_path / "outputs"
    output_link.symlink_to(data, target_is_directory=True)

    report = storage.preflight_storage(
        output_link / "run",
        required_root=data,
        minimum_free_gib=0,
    )

    assert report.resolved_output_dir == str(data / "run")
    assert report.required_root == str(data)
