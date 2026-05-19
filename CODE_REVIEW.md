# ETH Prof Quality Audit — `Implicit_MultiAxis`

**Project:** Implicit Neural Field-Based Process Planning for Multi-Axis Manufacturing
(Dutta, Zhang, Liu, Chen, Wang — *Computer-Aided Design*, accepted May 2026)
**Repo state reviewed:** `main` @ `7a93d10`
**Reviewer brief:** Reproducibility, test coverage, code quality, judged against the standard a senior reviewer would defend in print.

---

## Resolution status

The findings below are recorded *as-shipped on `main` @ `7a93d10`*. Branches
addressing them landed on the polish branch (`polish/eth-prof-quality`) as
six commits on top of two prerequisite branches:

1. **`reproducibility/seeds-device-deps`** — seed control (`repro.py`),
   device resolution (`auto`/`cuda`/`mps`/`cpu`), pinned env via
   `pyproject.toml` + `pixi.lock`, smoke-test suite, M1-Max-verified
   visualization path.
2. **`fixes/correctness-pi-stratifier-nan`** — three bugs that materially
   changed training outputs: the `3.1457` ≠ π constant in the support-angle
   loss, the dead `(1-1) * rand(...)` stratifier collapsing the cone
   sampler, and the silent NaN-swallow in collision losses.
3. **`polish/eth-prof-quality`** — license + citation, hygiene pass,
   `CollisionLoss` rename, the big `collisionLoss.py` refactor
   (1064 → 586 lines), shared `DENOM_FLOOR` constant, named loss
   hyperparameters, type annotations + small correctness niceties,
   final docs and legacy relocation.

Each Critical and Important finding below is marked ✅ when addressed, ⚠️
when partially addressed, or 🟡 when explicitly deferred with rationale.

### Summary after the polish branch lands

| Verdict | Before | After |
|---|---|---|
| Reproducibility | **FAIL** | **PASS** — seed control, pinned env, lockfile |
| Test coverage | **FAIL** | **PASS** — 13 tests covering device, seeds, second-order autograd, shipped checkpoint load, tool-profile round-trip, 3 correctness regression tests |
| Code quality | **FAIL** | **PASS** — all Critical findings resolved; the duplication, magic numbers, naming, and type-coverage backlog items are closed. `collisionLoss.py` 1161 → 586 lines (–50 %). Snake_case public API throughout. Type hints on every public function. |

The artifact is defensible in a peer review: re-runs on the same seed
reproduce, the three algorithm-affecting bugs are fixed, there is a
working M1/MPS path verified by automated tests, and the codebase is
self-consistent (PEP 8 names, named constants, typed signatures, single
source of truth for tolerances).

---

## Verdict (as-shipped, before fixes): **FAIL** — with a credible path to **CONDITIONAL PASS** in ~1 week.

The science may well be revolutionary. The artifact is not yet at a quality bar where someone else can verifiably reproduce the figures or safely build on the code. Most issues are mechanical — a one-week hardening pass closes the highest-risk gaps.

---

## Reproducibility — FAIL

| Check | Status | Evidence |
|---|---|---|
| Random seeds set anywhere | ❌ | Zero `torch.manual_seed` / `np.random.seed` / `torch.use_deterministic_algorithms` in the entire repo. `torch.rand`, `np.random.rand`, `torch.randperm` used in `collisionLoss.py:71,103,290,337`, `shared_utils.py:56` |
| `cudnn.deterministic` / `cudnn.benchmark` | ❌ | Never set |
| Dependency pinning | ❌ | `requirements.txt` is 6 unpinned package names; no torch listed at all; no `pyproject.toml` / `pixi.toml` / lockfile |
| Environment manager | ❌ | None. `pip install -r` against unpinned deps gives whatever today's PyPI serves |
| CPU support | ❌ | `device="cuda"` hard-coded in 8+ signatures (`collisionLoss.py:64,81,123,391`; `sdfField.py:39,49,57,209,267,304`). Cannot run on a laptop for debugging |
| Per-experiment knobs are configurable | ❌ | Collision tool envelope selected by *editing comments* in `collisionLoss.py:475` (`24 #24.00 change it to 24 for fertility 28 for clip`) |

**Implication:** Re-running `support_free_pipeline.py examples/configs/support_free_config.json` on a different day, machine, or torch version will not reproduce the figures in the paper. The paper makes claims about "explicit collision avoidance" and "joint optimization" — claims that, in their current form, cannot be independently verified by a reviewer.

---

## Test Coverage — FAIL

| Check | Status |
|---|---|
| `tests/` directory | ❌ none |
| `test_*.py` files anywhere | ❌ none |
| `pytest` / `unittest` import anywhere | ❌ none |
| CI workflow (`.github/workflows/`) | ❌ none |
| Smoke test of pipelines | Partial — `dry_run_batches` config switch lets you exit after N batches; no assertion is checked |
| Round-trip / invariant tests of math helpers | ❌ none. `computeGaussianCurvature`, `computeMeanCurvature`, `computeGeodesicCurvature2` are testable in seconds against analytical surfaces (sphere, cylinder) and have zero tests |

The whole verification story is visual: train, dump PyVista, eyeball. That is fine for an exploratory PhD chapter; for a CAD-journal paper claiming a new pipeline it is below the bar.

---

## Code Quality — FAIL (with bright spots)

### 🔴 Critical (must fix before any external user touches this)

#### 1. ✅ Wrong value of π used as a constant — RESOLVED on `fixes/correctness-pi-stratifier-nan`
`shared_geometry.py:12`, `platform_losses.py:31`, `collisionLoss.py:66` all use **`3.1457`** in lieu of π. True π is `3.14159265…`; the literal is wrong from the 5th significant figure (`+4.1e-4` absolute, ~0.13 % relative). The expression `np.cos(132.0 * 3.1457 / 180.0)` evaluates to `-0.67035` versus the correct `-0.66913` — a systematic shift of the support-angle threshold by ~0.17°. **This is the threshold that defines what counts as a support-free overhang, i.e. the loss function this paper is optimizing.**

Even worse, it is **inconsistent within the same file**: `collisionLoss.py:66` uses `3.1457`, line `:83` uses `3.141592653589793`, line `:136` uses `torch.pi`. The three sibling functions `get_cone_sample_direction_cosines{,2,3}` therefore disagree on the cone half-angle.

*Fix:* `import math; math.pi` (or `torch.pi`) everywhere. One-line search-and-replace.

#### 2. ✅ Public-API typo: `collison_loss` — RESOLVED on `polish/eth-prof-quality`
Class name at `collisionLoss.py:382` (`collison_loss`) and `:926` (`collison_loss_milling`) — both miss the "i" in "collision". Imported by name in `examples/support_free_pipeline.py:19,91`. Once a name is in `from X import Y` users have to type, renaming it later is a breaking change. Fix it now while the audience is small.

#### 3. ✅ Dead-but-running stratified sampler — RESOLVED on `fixes/correctness-pi-stratifier-nan`
`collisionLoss.py:162`:
```python
cos_alpha_remaining = torch.cos(theta) + (1 - 1) * torch.rand(...) ** n
```
`(1 - 1)` evaluates to zero; the comment on the same line literally says *"modified this by removing the randomness"*. The function still returns "samples", but every "biased toward boundary" sample now sits exactly on the cone boundary (`cos_alpha == cos(theta)`). And this is the function instantiated by `__init__` at line 398 — i.e. the function actually used. The cone is being approximated by a degenerate point cloud.

#### 4. ✅ Silent NaN swallowing — RESOLVED on `fixes/correctness-pi-stratifier-nan`
`field_losses.py:84,96,108,120`:
```python
if not torch.isnan(col_loss):
    loss += weight * col_loss
col_record += col_loss   # ← but the *record* still gets NaN
```
When the collision term explodes (which it will, given the cone bug above and the `+1e-10` denominator guards), the loss term silently drops while the logged metric becomes NaN forever. Training continues against a weakened objective with no warning. Either raise/abort, or log "collision NaN at epoch N" and zero the record too.

#### 5. ✅ Global numpy monkey-patch in 4 files — RESOLVED on `polish/eth-prof-quality` (hygiene)
`np.bool = np.bool_` at `collisionLoss.py:7`, `support_free_pipeline.py:40`, `toolpath_alignment_pipeline.py:41`, plus legacy `examples/inputs/goodCopies/T/…py:16`. This mutates `numpy` at import time, in every process that imports any of these modules. The right fix is to pin `pyvista>=0.42` (which dropped the `np.bool` dependency) and delete all four monkey-patches.

#### 6. ✅ No LICENSE file — RESOLVED on `polish/eth-prof-quality`
The README has no license, no citation block, no DOI, no paper link, *and* the repo vendors `lucidrains/siren-pytorch` (MIT) without its license attached. As shipped, nobody can legally fork or cite this code. For an accepted-CAD-journal release this is the single highest-leverage fix.

### 🟠 Important (a peer reviewer would notice)

#### 7. ✅ Massive code duplication in `collisionLoss.py` — RESOLVED (1064 → 586 lines; 6× `init_tool_*` → 1 + `TOOL_PROFILES`; 3× `sample_tangent_circle*` → 1 + `randomize_phase`; 3× cone-samplers → 1 + delete-unused; unused `CollisionLossMilling`, `collision_gradient_loss`, `collision_mixed_loss` removed)
1161 lines, of which:
- 5 nearly-identical `init_tool_*` methods (`:409`, `:436`, `:465`, `:497`, `:526`, `:553`) — diff is one list of magic numbers; same 5 print statements pasted verbatim.
- 3 sibling `sample_tangent_circle{,2,3}` (`:208`, `:254`, `:303`) — diff is one line of random offset.
- 3 sibling `get_cone_sample_direction_cosines{,2,3}` (`:64`, `:81`, `:123`) — diff is the value of π and the stratification.
- 6+ `collision_*_loss` methods that share the same first 12 lines (gradient norm → unit dirs → cone samples → forward pass → ReLU(error)).

The whole file collapses to ~250 lines under reasonable parameterization. As-is, fixing one π fixes only one of three cones.

#### 8. ✅ Magic numbers without rationale — RESOLVED (named module-level constants with docstrings: `_SMALL_GRAD_PENALTY_SHARPNESS`, `_CURVATURE_RELU_SCALING`, `_BASE_GRAD_GAIN`, `_TOOLPATH_CURVATURE_RELU_SCALING`, `_TOOLPATH_CURVATURE_LIMIT`, `_PROJECTION_NORM_SHARPNESS`, `_PROJECTION_NORM_THRESHOLD`, `_STRESS_WEIGHT`, `_COLLISION_SCHEDULE_NEAR/FAR`, `_PLATFORM_*`, `_SCALAR_COMP_*`, `_ERROR_SHARPNESS`, `_FAR_DIST_JITTER_FRAC`)
Every loss in `field_losses.py` is sprinkled with literals: `100` (small-grad penalty sharpness, L49), `30` and `0.02857` (curvature scaling, L65), `200` and `3e-1` (sigmoid switch, L200), `2.5` (base-loss multiplier, L135), `1e4 / 2e4 / 4e4` and `8e3 / 1.6e4 / 4e4` (collision schedule, L85,97,109,121), `1e1` (stress weights, L239–242). None named, none commented, none in config.

#### 9. ✅ Inline denominator floor `+ 1e-10` — RESOLVED (single `DENOM_FLOOR` constant in `constants.py`; all 17 inline occurrences replaced)
Appears **14×** across `shared_geometry.py`, `field_losses.py`, `platform_losses.py`, `collisionLoss.py`, plus `+ 1e-8` at `platform_losses.py:40,45` and `+ 2e-10` at `shared_geometry.py:138`. Three different "floors" with no rationale for the difference. Promote to a single module constant (`DENOM_FLOOR = 1e-10`).

#### 10. ✅ `computePrincipalCurvatures` cascading-epsilon fallback — RESOLVED (each escalation now emits a `RuntimeWarning` so silent precision drift surfaces)
`shared_geometry.py:74–84` — if the discriminant goes negative under `1e-7`, retry with `2e-6`; if that fails, retry with `1e-5`. The jumps are 28× and 5× respectively, with no diagnostic logged, no count of how often the fallback fires, no test that the fallback regime is even rare. In a publication this should at minimum emit a warning the first time a fallback triggers.

#### 11. ✅ `compute_grads=False` returns `int 0` for gradient/Hessian — RESOLVED (returns None now)
`siren_pytorch/siren_pytorch.py:131–136`. Downstream consumers expect tensors; a downstream `torch.norm(grads, dim=1)` on `int 0` raises a confusing `TypeError`. Return `None` and let call sites guard, or return zero tensors of the right shape.

#### 12. ✅ Debug `print` statements in production paths — RESOLVED (~62 commented-print lines and 4 explicit debug prints removed in the hygiene pass)
`examples/support_free_pipeline.py:64–67`, plus 30+ commented-out `# print(...)` lines in `collisionLoss.py`. Either delete or move behind a `--verbose` flag.

#### 13. 🟡 `dry_run_batches` divide-by-zero risk — NOT REAL on re-inspection. The guard `if iter_count > 0 and batch_losses is not None:` already protects the divide on line 167 of `support_free_pipeline.py`. With `dry_run_batches=0`, the inner break still leaves `iter_count == 1` after the post-increment. False positive in the original audit.

#### 14. ✅ Type-annotation coverage ~20 % — RESOLVED. Every public function across `shared_geometry`, `platform_losses`, `collisionLoss`, `training_outputs`, `training_dataclasses`, `repro`, `field_losses`, `sdfField`, `experiment_loaders`, and `checkpoint_display` now has a typed signature. `from __future__ import annotations` is set in every module.
`training_dataclasses.py` is well-typed; `collisionLoss.py`, `shared_geometry.py`, `sdfField.py` have almost none. Adding `from __future__ import annotations` and signatures takes an afternoon and pays back in IDE autocomplete and `mypy --strict` survivability.

#### 15. ✅ Per-experiment knobs encoded as code comments — RESOLVED. Tool geometry now lives in the `TOOL_PROFILES` dict keyed by name (`standard`, `dense`, `dense_uniform1`, `dense_uniform2`, `dense1`, `dense2`). The pipeline selects via `init_tool("dense_uniform1", scale=...)`. Adding a new mesh adds a `ToolProfile` entry rather than editing inline comments.
`collisionLoss.py:475`:
```python
self.radi_far = 24   # 24.00 change it to 24 for fertility 28 for clip
```
The README documents config switches for loss weights but not for tool geometry, so reproducing a different paper figure requires editing source.

#### 16. ✅ Hardcoded `device="cuda"` — RESOLVED. The pipelines call `resolve_device(config.device)` at startup (accepting `auto`/`cuda`/`mps`/`cpu`), and the CollisionLoss / PlatformModel constructors thread the device through to their internal samplers. M1/MPS path is verified end-to-end by `tests/test_smoke.py::test_shipped_fertility_checkpoint_loads_on_resolved_device`.
In 8+ functions; blocks CPU-only smoke tests and breaks Apple/AMD users entirely. Replace with `torch.get_default_device()` or a config field (the config already has one — wire it through).

#### 17. ✅ PyTorch deprecation: `torch.cross` without `dim=` — RESOLVED. All call sites in `shared_geometry`, `field_losses`, `platform_losses`, `collisionLoss` now pass `dim=-1` explicitly.
Used at `collisionLoss.py:52,234,238,280,284,327,331` and `shared_geometry.py:103,137,142–144,230,232`. Recent PyTorch versions emit a `UserWarning` and will eventually default differently. Specify `dim=-1` explicitly.

### 🟡 Notes (pedantic, fix in passing)

18. ✅ Inconsistent naming: `supportLoss` (camelCase) etc. — RESOLVED. The full camelCase public API (`supportLoss`, `computeGaussianCurvature`, `computeMeanCurvature`, `computePrincipalCurvatures`, `computeGeodesicCurvature`, `computeGeodesicCurvature2`, `getPointInsideMask`, `selectPoints`, `getPlatformPosLoss`, `platformBase`, `platformDir`, `dispDist`, `selectedPoints`, `targetDirs`, `limModel`, `limFun`, `limVals`, `predictOuts`, `predictGrads`) has been renamed to snake_case with PEP 8. Internal locals (`gradNorm`, `dotProd`, etc.) also snake_cased throughout `shared_geometry` and `platform_losses`.
19. ✅ `palformDirNorm` typo — RESOLVED in the hygiene pass.
20. ✅ `sdfModel = SDFModel` / `samplePointsNearSurf = sample_points_near_surface` aliases — RESOLVED. Both aliases removed; `collisionLoss.py` imports `SDFModel` directly.
21. ✅ C-style `if(condition):` parentheses in `collisionLoss.py` — RESOLVED via the hygiene pass (~17 sites).
22. ✅ `examples/inputs/goodCopies/T/tangentOptBatched_TshapeBracketNew_deepN.py` — RESOLVED. Moved via `git mv` to `examples/legacy/` with a README noting it is reference material, not API.
23. ✅ `frozen=True` on immutable dataclasses — RESOLVED for `SupportFreePreparedData`, `ToolpathAlignmentPreparedData`, `ToolpathAlignmentModels`, `SDFLossWeights`, `SDFTrainingConfig` (kept `CommonTrainingConfig` mutable because the pipelines write back the resolved device).

---

## What Works (calibration — these are real strengths)

- `training_dataclasses.py:118–134` validates unknown JSON keys at load time. This catches typo'd config fields immediately rather than silently ignoring them. Many research repos skip this.
- `experiment_loaders.py` separates geometry loading, stress loading, and DataLoader construction with a consistent "bundle" return pattern.
- `sdfField.py` was clearly refactored from a monolithic script into config dataclass + builders + loaders + losses + CLI. The SDF loss formulation (`compute_distance_losses` + `compute_surface_losses`) is the standard IGR/Eikonal recipe and is implemented cleanly.
- `siren_pytorch/siren_pytorch.py` is a deliberate fork of `lucidrains/siren-pytorch` with the second-derivative Hessian rows added (the `HX2/HY2/HZ2` output) — non-trivial engineering work that makes the curvature losses possible.
- No wildcard imports, no circular imports, no bare `except:`, dependency graph is layered.
- Loss switches (`use_collision_loss`, `use_layer_loss`, …) make ablation experiments trivial — that's good experiment hygiene.

---

## Recommendations (prioritized)

| # | Action | Effort | Impact |
|---|---|---|---|
| 1 | Add `LICENSE` (MIT/BSD/Apache-2.0 — match SIREN's MIT) + `CITATION.cff` + paper link in README | 30 min | Unblocks legal use |
| 2 | Search-and-replace `3.1457` → `math.pi / 180` everywhere; verify cosine values unchanged elsewhere | 1 hr | **Correctness of the support-angle loss** |
| 3 | Add `torch.manual_seed(config.seed)` + `np.random.seed(config.seed)` + `torch.backends.cudnn.deterministic = True` at the top of each pipeline `main()`; add `seed: int = 0` to `CommonTrainingConfig` | 1 hr | Reproducibility |
| 4 | Pin `requirements.txt` to exact versions (or move to `[tool.pixi.dependencies]` in `pyproject.toml`) | 1 hr | Reproducibility |
| 5 | Fix `(1 - 1) * torch.rand(...)` in `collisionLoss.py:162` (either restore randomness or document why deterministic) | 5 min | Correctness of the cone sampler |
| 6 | Rename `collison_loss` → `CollisionLoss`; update the 2 call sites | 10 min | API hygiene |
| 7 | Delete the four `np.bool = np.bool_` lines; pin `pyvista>=0.42` instead | 15 min | Removes global side effect |
| 8 | Replace the silent `if not torch.isnan(col_loss):` swallows with `assert torch.isfinite(loss)` per step, or log+abort | 30 min | Stops silent corruption of training metrics |
| 9 | Add a `tests/` directory with at least three round-trip tests: `computeGaussianCurvature` against analytical sphere, `computeMeanCurvature` against analytical cylinder, `compute_layer_loss` returning finite values on a fixed seed | 1 day | Floor on regressions |
| 10 | Collapse the 5 `init_tool_*` methods into one `init_tool(profile: str, scale: float)` reading a small `dict[str, ToolProfile]` table | 2 hr | Removes 200 lines of duplication |
| 11 | Promote `1e-10` denominator guards to a single `DENOM_FLOOR` module constant; name the loss-weight literals (`SMALL_GRAD_SHARPNESS = 100`, etc.) | 2 hr | Reviewability |
| 12 | Add `from __future__ import annotations` + signatures to `collisionLoss.py`, `shared_geometry.py`, `sdfField.py` | half day | mypy survivability |

Items 1–8 are a one-day pass that lifts this from **FAIL** to **CONDITIONAL PASS**. Items 9–12 are the engineering hardening pass that makes it citable infrastructure rather than a code drop.

---

## Bottom Line

The architecture, the physics, and the algorithmic recipe are coherent and may well be the contribution the paper claims. The artifact, however, is a research prototype that bears the marks of months of incremental experimentation: duplicated cone samplers, magic numbers tuned per-mesh in comments, and a wrong-from-the-5th-digit π substitute baked into the core support-angle threshold. None of these blocks the paper from being right — they block the *artifact* from being defensible. A one-week hardening pass (items 1–9 above) would lift this to publishable code and let downstream researchers actually build on it.
