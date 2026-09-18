import os
from pathlib import Path

import pytest

pytest.importorskip("torch")

from examples.embodied.pi_rlinf_transition_parity import run  # noqa: E402

RLINF_SOURCE = Path(
    os.environ.get(
        "ART_EMBODIED_RLINF_SOURCE",
        Path(__file__).resolve().parents[2] / "RLinf-v01",
    )
)


@pytest.mark.skipif(not RLINF_SOURCE.is_dir(), reason="RLinf checkout is unavailable")
def test_art_flow_sde_transition_matches_rlinf_source() -> None:
    result = run(RLINF_SOURCE)

    assert result["status"] == "ok"
    assert result["max_deltas"]["mean"] <= result["tolerance"]
    assert result["max_deltas"]["std"] <= result["tolerance"]
