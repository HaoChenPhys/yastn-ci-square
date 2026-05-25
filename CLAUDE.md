# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal, self-contained demo that uses [yastn](https://github.com/yastn/yastn)
**directly** (no wrapper code — only `yastn` is imported) to load a pre-converged
Chern-insulator (CI) iPEPS / CTM environment on the square lattice and measure
observables. The CI states are `Z2`-symmetric spinless fermions, D=4, on a 2×2
`RectangularUnitcell`; the physics model is the chiral t-V checkerboard model
(see the docstring in `examples/03_measure.py`).

## Setup & running

`yastn` lives as a git submodule pinned to the **`oe_blocksparse` branch** (not
`main`). This matters: `examples/04_measure_nn_scratch.py` calls
`yastn.contract_with_unroll` / `yastn.get_contraction_path`, which only exist on
that branch. Do **not** run `git submodule update --remote yastn` expecting it to
track the right branch — verify with `git -C yastn branch --show-current`.

```bash
git submodule update --init --recursive   # if yastn/ is empty
pip install -e yastn                        # or: export PYTHONPATH="$PWD/yastn"
```

Scripts are run from inside `examples/` (they use bare `import _jsonio` and
`__file__`-relative paths) and have a **run-order dependency** via the
git-ignored `examples/out/` directory:

```bash
cd examples
python 00_yastn_basics.py   # optional primer (tensor init + contraction)
python 01_build_and_save.py # writes out/Z2_..._chi_128_state.json
python 02_load_and_ctm.py   # writes out/ci_env_chi.json (+ round-trip check)
python 03_measure.py        # reads BOTH out/ files written above
python 04_measure_nn_scratch.py  # reads out/ci_env_chi.json
```

`03` and `04` fail if `01`/`02` haven't been run, because they read from
`out/`, not from `CI_states/`.

## Architecture / things to know

- **One shared config, everywhere.** Every script rebuilds the same config:
  `yastn.make_config(sym="Z2", fermionic=True, default_dtype="complex128")`
  (numpy backend). Passing it to `yastn.from_dict` deliberately overrides
  whatever backend/device the JSON was written with (the fixtures came from a
  torch backend). Operators (`yastn.operators.SpinlessFermions`) must be built
  with the **same** sym/backend/dtype, or block structures won't match in the
  `measure_*` contractions.

- **`examples/_jsonio.py` exists because yastn `to_dict` payloads don't survive
  a plain `json.dump` round-trip.** It adds: a `NumPy_Encoder` (ndarrays →
  lists, complex → `{"real","imag"}`), a `complex_decoder`, and a critical
  `_fix_int_keys` post-load pass — yastn's `site_data` uses **int** keys (from
  `Geometry.site2index`) that JSON silently coerces to strings. The encoder
  pair is kept byte-compatible with `tn-torch_dev_square`'s `tensor_io.py`.

- **Fixtures in `examples/CI_states/`.** The `_state.json` (the PEPS payload) is
  chi-independent — the `chi_128` in its filename just records the chi it was
  produced at. Only the **chi=32** `_state_ctm_env.json` is committed; the
  chi=64 and chi=128 env files are git-ignored (large, regenerable). Scripts
  `02`/`03` use the chi=32 env.

- **Script 04 is the deep one.** It reconstructs a horizontal NN
  `<c⁺c>` measurement from scratch by feeding the full 14-tensor double-layer
  network (ket/bra kept separate, env edge legs un-fused) to a single
  `contract_with_unroll` call, with fermionic swap gates passed explicitly.
  Its module docstring contains the full contraction diagram and the swap-gate
  bookkeeping that yastn normally hides inside `DoublePepsTensor.fuse_layers`.
  It asserts the result matches `EnvCTM.measure_nn` to 1e-12.

- The energy port in `03_measure.py:energy_per_site` mirrors
  `tV_checkerboard_model.energy_per_site` from the (separate) `tn-torch_dev_square`
  codebase; reference value `E/site = -1.0217145251` at chi=128.
