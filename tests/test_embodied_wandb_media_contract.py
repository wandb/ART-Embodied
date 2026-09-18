import hashlib
from pathlib import Path

import pytest

from art_embodied.wandb_media_contract import (
    VideoExpectation,
    _decode_video,
    _parse_expectation,
    verify_run_videos,
)


class FakeRun:
    url = "https://wandb.ai/entity/project/runs/test"

    def __init__(self, rows, content=b"video"):
        self.rows = rows
        self.content = content
        self.scans = 0
        self.downloads = []

    def scan_history(self):
        self.scans += 1
        return iter(self.rows)

    def file(self, name):
        run = self

        class File:
            def download(self, *, root, replace):
                run.downloads.append(name)
                path = Path(root) / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(run.content)
                return path.open("rb")

        return File()


def metadata():
    return {
        "_type": "video-file",
        "path": "media/videos/test.gif",
        "sha256": hashlib.sha256(b"video").hexdigest(),
    }


def audit(run, tmp_path, *, count=1, decoder=None):
    return verify_run_videos(
        run,
        [VideoExpectation(5, "media/simulation/eval", count)],
        download_root=tmp_path,
        decode_video=decoder or (lambda _: {"frames": 10, "width": 20, "height": 20}),
    )


def row(update=5, **entries):
    return {"experiment/update": update, **entries}


KEY = "media/simulation/eval/0"


@pytest.mark.parametrize("native_step,verified", [(5, True), (10, False), (6, False)])
def test_native_media_step_must_match_update(tmp_path, native_step, verified):
    run = FakeRun([row(5, **{KEY: metadata(), "_step": native_step})])
    report = verify_run_videos(
        run,
        [VideoExpectation(5, "media/simulation/eval", 1)],
        download_root=tmp_path,
        decode_video=lambda _: {"frames": 10, "width": 20, "height": 20},
        require_native_steps=True,
    )
    assert report["media_verified"] is verified


def test_current_attachment_checked_and_no_app_claim(tmp_path):
    run = FakeRun([row(4, **{KEY: metadata()}), row(5, **{KEY: metadata()})])
    report = audit(run, tmp_path)
    assert report["media_verified"]
    assert not report["app_rendering_verified"]
    assert run.scans == 1
    assert len(run.downloads) == 1


def test_old_video_count_cannot_satisfy_current_update(tmp_path):
    run = FakeRun([row(0, **{KEY: metadata()})] * 20)
    report = audit(run, tmp_path)
    assert not report["media_verified"]
    assert report["expectations"][0]["errors"][0]["kind"] == "missing"
    assert not run.downloads


def test_duplicate_attachment_fails_even_if_identical(tmp_path):
    run = FakeRun([row(**{KEY: metadata()})] * 2)
    report = audit(run, tmp_path)
    assert not report["media_verified"]
    assert report["expectations"][0]["errors"][0]["kind"] == "duplicate"


def test_checks_all_numbered_keys_not_just_count(tmp_path):
    run = FakeRun([row(**{KEY: metadata(), "media/simulation/eval/2": metadata()})])
    report = audit(run, tmp_path, count=2)
    assert not report["media_verified"]
    assert {e["kind"] for e in report["expectations"][0]["errors"]} == {
        "missing",
        "unexpected_attachment",
    }


@pytest.mark.parametrize(
    "field,value", [("path", "../secret"), ("sha256", "bad"), ("_type", "image-file")]
)
def test_invalid_metadata_rejected_before_download(tmp_path, field, value):
    item = {**metadata(), field: value}
    run = FakeRun([row(**{KEY: item})])
    assert not audit(run, tmp_path)["media_verified"]
    assert not run.downloads


def test_corrupt_bytes_fail(tmp_path):
    run = FakeRun([row(**{KEY: metadata()})], content=b"corrupt")
    assert not audit(run, tmp_path)["media_verified"]


def test_wrong_declared_size_fails(tmp_path):
    run = FakeRun([row(**{KEY: {**metadata(), "size": 999}})])
    assert not audit(run, tmp_path)["media_verified"]


def test_existing_symlink_cannot_escape_download_root(tmp_path):
    download_root = tmp_path / "downloads"
    download_root.mkdir()
    (download_root / "media").symlink_to(tmp_path, target_is_directory=True)
    run = FakeRun([row(**{KEY: metadata()})])
    assert not audit(run, download_root)["media_verified"]
    assert not run.downloads


def test_decoder_failure_is_not_hidden(tmp_path):
    def fail(_):
        raise ValueError("truncated video")

    report = audit(FakeRun([row(**{KEY: metadata()})]), tmp_path, decoder=fail)
    assert not report["media_verified"]
    assert "truncated video" in report["expectations"][0]["errors"][0]["error"]


def test_separate_versions_and_sparse_history_rows(tmp_path):
    run = FakeRun(
        [
            row(4, **{"media/simulation/train/0": metadata()}),
            row(5),
            row(5, **{KEY: metadata()}),
        ]
    )
    report = verify_run_videos(
        run,
        [
            VideoExpectation(4, "media/simulation/train", 1),
            VideoExpectation(5, "media/simulation/eval", 1),
        ],
        download_root=tmp_path,
        decode_video=lambda _: {"frames": 1, "width": 20, "height": 20},
    )
    assert report["media_verified"]
    assert run.scans == 1


def test_decode_entire_real_gif(tmp_path):
    pytest.importorskip("av")
    image = pytest.importorskip("PIL.Image")
    path = tmp_path / "video.gif"
    image.new("RGB", (8, 8), "red").save(
        path,
        save_all=True,
        append_images=[image.new("RGB", (8, 8), "blue")],
        duration=100,
    )
    assert _decode_video(path) == {"frames": 2, "width": 8, "height": 8}


def test_expectation_validation():
    assert _parse_expectation("5:media/simulation/eval:8") == VideoExpectation(
        5, "media/simulation/eval", 8
    )
    for args in [
        (True, "prefix", 1),
        (-1, "prefix", 1),
        (0, "prefix/", 1),
        (0, "prefix", 0),
    ]:
        with pytest.raises(ValueError):
            VideoExpectation(*args)


@pytest.mark.parametrize(
    "size,valid",
    [
        (5, True),
        (5.0, True),
        (5.5, False),
        (float("nan"), False),
        (float("inf"), False),
        ("5", False),
        (True, False),
        (False, False),
    ],
)
def test_numeric_byte_count_is_exact_not_python_integer_only(tmp_path, size, valid):
    run = FakeRun([row(**{KEY: {**metadata(), "size": size}})])
    assert audit(run, tmp_path)["media_verified"] is valid


def test_boolean_cannot_masquerade_as_one_byte(tmp_path):
    item = {**metadata(), "size": True, "sha256": hashlib.sha256(b"x").hexdigest()}
    run = FakeRun([row(**{KEY: item})], content=b"x")
    assert not audit(run, tmp_path)["media_verified"]
