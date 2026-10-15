# AGENTS.md — AI Agent Guidelines for fVDB

This file provides persistent instructions for AI coding agents (Cursor, Copilot,
Codex, etc.) working on this codebase. Human contributors should also refer to
[CONTRIBUTING.md](CONTRIBUTING.md).

## Git Commits

- All commits **must** include a DCO sign-off (`git commit --signoff` / `-s`).
- All commits **must** be SSH-signed (`git commit --gpg-sign` / `-S`).
- **Never** skip hooks or signing: do not use `--no-verify` or `--no-gpg-sign`.

## General Style

- Avoid excessive use of emoji and non-ascii characters in code and documentation except
  where it is helpful to user experience or user interfaces.

## Python Code Style

- Sort imports and format Python code with **ruff** 0.15.7, the version pinned in CI, from the
  repository root:

  ```
  ruff check --select I --fix .
  ruff format .
  ```

- Ruff reads its settings (line length 120, target `py311`, and exclusions) from
  `[tool.ruff]` in `pyproject.toml`, so no extra flags are needed.
- `./build.sh format` runs clang-format, Ruff import sorting, and `ruff format`;
  `./build.sh format check` reports violations without rewriting files.

## C++ Code Style

- Format C++ source files under `src/` with **clang-format 18** using the
  repository's `.clang-format` style file:

  ```
  clang-format -i --style=file src/**/*.{h,cpp,cc,cu,cuh}
  ```

- Applicable extensions: `.h`, `.cpp`, `.cc`, `.cu`, `.cuh`.

## License Headers

Every source file **must** include the Apache-2.0 SPDX identifier.

Python files:

```python
# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
```

C++ files:

```cpp
// Copyright Contributors to the OpenVDB Project
// SPDX-License-Identifier: Apache-2.0
```

## Whitespace

- No trailing whitespace (CI enforces this; `.wlt` files and `wip/` are excluded).
- Use spaces, not tabs (binary and a few config files are excluded from this check).

## Testing

- Run relevant tests before pushing. Tests **must** be run from the `tests/`
  directory so that the installed `fvdb` package (with compiled C++ extensions)
  is imported instead of the unbuilt source tree at the repo root:

  ```
  cd tests && pytest unit -v
  ```

- To run the C++ gtest suite:

  ```
  ./build.sh ctest
  ```

- The `wip/` directory is excluded from test collection (see `norecursedirs` in
  `pyproject.toml`).

## Opening Issues

- Reference any related issues or PRs.
- Set appropriate labels (e.g. `bug`, `enhancement`, `documentation`).
- For bugs: provide clear reproduction steps, expected vs actual behavior, and
  environment details.

## Opening Pull Requests

- Reference the issue being fixed (e.g. "Fixes #NNN").
- Include a test plan with specific commands to verify the change.
- Ensure all CI checks pass before requesting review (DCO, codestyle, tests).
- Keep PRs focused on a single concern.
- Set appropriate labels matching the linked issue.

**Do not commit unless directed:**
- `.vscode/settings.json` - local IDE settings
- `.cursor/` - local Cursor config
- Test-specific matrices or configs with hardcoded paths

## CUDA Performance Work

- Prefer small, self-contained experiments within one operator. Change one
  performance mechanism at a time so captures can attribute the result.
- Do not build, run tests, or format unless requested. Report which validation
  was actually performed.
- The primary objective is multi-GPU scaling relative to single-GPU performance.
  Distinguish absolute kernel improvements from scalability improvements.
- Consider both one camera per GPU and multiple GPUs sharing one camera.

## Profiling and Comparisons

- Use the user's requested baseline. Do not silently substitute the preceding
  capture.
- Record the source revision, outstanding changes, checkpoint, and profiling
  options associated with each capture. Ask when attribution is ambiguous.
- Compare matching iterations and exclude warmup consistently. State the sample
  count and exclusions.
- Report per-GPU kernel durations, start skew, combined elapsed span, and
  end-to-end iteration time separately.
- Check GPU clocks and competing graphics/compute activity before attributing
  timing differences to code. GPU1 graphics activity has affected prior runs.
  Treat activity-based filtering as supporting evidence, not a controlled run.
- Use page-fault-enabled captures primarily to locate and correlate faults;
  tracing overhead makes them unsuitable as direct performance baselines.
- Distinguish measured NVLink bytes from integrated utilization metrics.
  A copy labeled Peer-to-Peer does not by itself establish NVLink traffic.
- High LSU utilization does not imply DRAM bandwidth saturation. Use instruction,
  dependency-stall, cache, and bandwidth measurements to identify the bottleneck.
- Check registers, occupancy, and spills when changing reductions or unrolling.
  Fewer instructions do not guarantee lower runtime.
- Keep detailed capture comparisons and experimental outcomes in a profiling log,
  rather than embedding changing benchmark numbers in AGENTS.md.

## Multi-GPU Implementation Conventions

- The sleepKernel is intentional; do not propose removing it as an optimization.
- Name getCurrentCUDAStream() results `stream`, and streams obtained for
  prefetching `prefetchStream`.
- Prefer std::tie for assignment to existing variables and indexed loops when
  managing per-device streams and events.
- Set the appropriate CUDA device before destroying its events.
- Keep stream management internal to operators.
- Do not mix PyTorch kPrivateUse1 and kCUDA tensor backends.
- Preserve Gaussian ownership and camera/tile ownership explicitly when reasoning
  about placement. Prefetching an overlapping range onto multiple GPUs can cause
  migration contention.
- Avoid quadratic cross-device event waits. Add synchronization only for a
  concrete dependency; do not assume page-sharing hazards are proven absent.
- Use consistent index types and existing helpers such as ceil_div. Avoid casts
  that are unnecessary given the operand and destination types.
