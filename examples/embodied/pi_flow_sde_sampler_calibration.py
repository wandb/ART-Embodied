"""Backward-compatible PI entrypoint for the generic Flow-SDE calibration."""

import asyncio

from examples.embodied.flow_sde_sampler_calibration import (
    _diagnostic_config,
    parse_args,
    run,
)

__all__ = ["_diagnostic_config", "parse_args", "run"]


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
