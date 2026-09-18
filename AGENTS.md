# Repository Boundaries

- Keep internal research notes, operator diaries, incident reports and unpublished
  support drafts outside this repository, under `ART_EMBODIED_PRIVATE_DIR`.
  This directory must be outside every Git checkout and accessible only to its
  owner. Do not use archive branches or old commit links to publish these records.
- Public documentation contains reviewed results, technical guidance and
  reproduction instructions. Review staged content and links before committing.
- Install the local guards with `git config core.hooksPath .githooks`. Never bypass
  a failing guard to publish internal material. CI also checks repository history.
- Private paths and explicitly marked content are mechanically checked; these
  checks do not replace reviewing documents for unmarked internal information.
- Preserve qualified policy implementations and runtime settings when changing
  documentation or release controls. Keep operational evidence outside Git.
