# Dependency Profiles and Security

The current checkout includes dependency updates made after `0.1.0rc2`.
The existing RC2 release assets do not include all of these changes.
Use the checkout instructions in the [README](../../README.md) for the
updated profiles.

## Installation Profiles

| Installation | Dependency handling |
| --- | --- |
| Checkout with uv 0.12.0 or newer | `uv sync --locked` applies the security floors and the LiteLLM 1.101.0 / Diffusers 0.38.0 overrides in `pyproject.toml`. The lock retains the policy-specific Torch and Transformers versions. |
| Plain pip | `constraints/security.txt` applies minimum versions for selected dependencies. It does not apply uv's LiteLLM or Diffusers overrides. |
| GR00T N1.7 installer | Uses NVIDIA's separate runtime with Diffusers 0.38.0 and Safetensors 0.8.0. It does not apply the checkout's LiteLLM override or other security floors. |
| Other native environments | Follow their own installer and pinned dependencies. Changes to the root lockfile do not update them. |

The OpenVLA-OFT profile uses SentencePiece 0.2.1. The standard constraints
also update GitPython, Click, Fickling, aiohttp, Pillow, cryptography, and
hydra-core.
The overrides replace selected upstream requirements without modifying
upstream package source or metadata. Dependency-consistency tools may report
these intentional differences from the upstream requirements.

## Validation

- SentencePiece: tokenizer comparisons on 16,436 strings and an OpenVLA-OFT
  GPU comparison matched tokenization, actions, likelihoods, and gradients.
- LiteLLM: package, import, lifecycle, and regression tests cover the ART and
  older Tokenizers combinations used by the add-on. ART-Embodied does not
  directly use LiteLLM's serving APIs.
- Diffusers: fixed-input comparisons cover LeRobot diffusion policies and
  GR00T N1.7. A fresh GR00T policy environment completed one RoboCasa GRPO
  update, checkpoint restoration, and W&B metric, video, and artifact checks.

These checks establish bounded compatibility, not long-run learning
equivalence or a security guarantee. Updating this checkout does not change
previously installed environments or previously built release assets.

## Remaining Precautions

Some pinned dependencies still have known security advisories. This includes
policy-specific Torch and Transformers versions and dependencies in separate
native runtimes. The root lockfile is not an inventory of every worker
environment. Review the resolved packages for the profile you deploy.

- Use trusted checkpoints, tokenizer assets, datasets, repositories, and
  experiment configurations. Some loaders execute model code or restore
  Python-pickle optimizer and RNG state. File hashes do not make untrusted
  inputs safe.
- Keep training workers separate from public API services and use narrowly
  scoped credentials.
- Scan the actual environment and distribution artifacts before deployment.
  A passing pull-request check does not cover every existing environment.
