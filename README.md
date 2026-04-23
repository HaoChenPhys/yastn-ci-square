# yastn CI-square example

Minimal, self-contained demo of using [yastn](https://github.com/yastn/yastn)
directly to load a converged Chern insulator (CI) iPEPS / CTM environment on the square lattice
and measure observables from it. Only `yastn` is imported — no wrapper code.

For the full API reference and further examples, see the
[yastn documentation](https://yastn.github.io/yastn/index.html).

## yastn tensor basics

A `yastn.Tensor` is a block-sparse, symmetry-aware tensor. Each tensor lives
on a `yastn.make_config` (symmetry group, backend, dtype, fermionic flag).
Every leg carries a signature `s ∈ {+1, -1}` — conventionally `+1` is
ingoing, `-1` is outgoing — plus a list of charge sectors `t` and their
block dimensions `D`. Only blocks whose leg charges are consistent with the
group fusion rule are stored.

A runnable walkthrough of this section lives in
[`examples/00_yastn_basics.py`](examples/00_yastn_basics.py); the snippets
below are excerpted from it.

### Initialization

```python
import yastn

# Z2 symmetry, numpy backend (default), complex128 — matches the CI states.
config = yastn.make_config(sym="Z2", fermionic=True, default_dtype="complex128")

# Reusable leg definitions: two Z2 sectors (charge 0 and 1), each of dim 2.
leg_in  = yastn.Leg(config, s=+1, t=(0, 1), D=(2, 2))
leg_out = yastn.Leg(config, s=-1, t=(0, 1), D=(2, 2))

# Rank-3 tensor, signature (+1, -1, -1), filled with random values in every
# allowed block.
a = yastn.rand(config=config, legs=[leg_in, leg_out, leg_out])

# Alternative: build an empty tensor and set specific blocks by hand. Only
# charge-conserving blocks are allowed — for Z2 the sum of leg charges,
# weighted by signature, must vanish mod 2.
b = yastn.Tensor(config=config, s=(+1, -1, -1))
b.set_block(ts=(0, 0, 0), Ds=(2, 2, 2), val="rand")
b.set_block(ts=(1, 1, 0), Ds=(2, 2, 2), val="rand")
b.set_block(ts=(1, 0, 1), Ds=(2, 2, 2), val="rand")
b.set_block(ts=(0, 1, 1), Ds=(2, 2, 2), val="rand")
```

### Contraction

```python
# tensordot: contract a's leg 2 with b's leg 0. The contracted legs must
# have opposite signatures (one ingoing, one outgoing) — yastn raises if
# they don't, which is how symmetry violations get caught early.
#
# Use .conj() on a leg spec to flip its signature when you want to build a
# partner tensor that can be contracted back.
c_leg0 = leg_out.conj()                           # s=+1, partners with leg_out
c = yastn.rand(config=config, legs=[c_leg0, leg_out, leg_out])

abc = yastn.tensordot(a, c, axes=(2, 0))          # result is rank-4

# `@` is the shorthand for "last leg of LHS × first leg of RHS".
assert yastn.norm(abc - (a @ c)) < 1e-12

# ncon / einsum give explicit index-label control when you need it.
abc_ncon = yastn.ncon([a, c], ((-0, -1, 1), (1, -2, -3)))
abc_eins = yastn.einsum("ijx,xkl->ijkl", a, c)

# Fully-contracted scalar (tensor of rank 0) → backend element → python scalar.
s = yastn.vdot(a, a).to_number()
```

The CI-square example below builds on exactly this machinery: the PEPS on-site
tensors and the CTM environment tensors are `yastn.Tensor`s on a Z2 fermionic
config, and all observables reduce to `tensordot`-style contractions that yastn
dispatches block-by-block.

## What it shows

The starting point is a folder of pre-converged CI fixtures under
`examples/CI_states/`, produced separately on a torch backend
(`Z2`-symmetric spinless fermions, D=4, 2x2 `RectangularUnitcell`):

- `Z2_..._state.json` — the standalone PEPS `Peps.to_dict` payload (chi-independent).
- `Z2_..._chi_{32,64,128}_state_ctm_env.json` — full `EnvCTM.to_dict` payloads
  (`psi` / `env` / `proj`) at each bond dimension chi.

All fixtures are JSON written via the ndarray-aware `_jsonio` helper (see
the "note on JSON" section below), so they round-trip through
`yastn.from_dict` unchanged. Script 01 reads the standalone state JSON;
script 02 reads the chi=32 env JSON.

1. **Load the state** (`01_build_and_save.py`) — read the standalone state
   JSON with `_jsonio.load` + `yastn.from_dict` under a numpy `Z2` config,
   and re-serialize it to JSON.
2. **Load the environment** (`02_load_and_ctm.py`) — rebuild the full
   `EnvCTM` from the chi=32 env JSON via `_jsonio.load` + `yastn.from_dict`,
   and re-serialize it to JSON. No CTMRG sweep is run; the env is already
   converged.
3. **Measure observables** (`03_measure.py`) — use
   `EnvCTM.measure_1site` / `EnvCTM.measure_nn` to get site occupations,
   nearest-neighbour hoppings, density-density correlators, and the energy
   of a spinless t-V Hamiltonian.

## Setup

### 1. Clone with the `yastn` submodule

`yastn` is pinned as a git submodule so the example always runs against a
known-good revision.

```bash
git clone --recurse-submodules <this-repo-url>
cd yastn-fci-square-example

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive

# To later pull in upstream yastn changes:
git submodule update --remote yastn
```

### 2. Create a Python environment

Yastn requires Python ≥ 3.10 plus `numpy`, `scipy`, `tqdm`, `h5py`, and
`opt_einsum`. Either a conda or a venv environment works.

```bash
# conda
conda create -n ci_square python=3.11 numpy scipy tqdm h5py opt_einsum -c conda-forge
conda activate ci_square

# OR: venv
python3.11 -m venv .venv
source .venv/bin/activate
pip install numpy scipy tqdm h5py opt_einsum
```

### 3. Make `yastn` importable

Install the submodule (recommended), or just point `PYTHONPATH` at it:

```bash
pip install -e yastn               # editable install of the bundled submodule
# or, no-install variant:
export PYTHONPATH="$PWD/yastn"
```

## Run

```bash
cd examples
python 00_yastn_basics.py    # optional primer: tensor init + contraction
python 01_build_and_save.py
python 02_load_and_ctm.py
python 03_measure.py
```

Round-trip artifacts (`ci_state.json`, `ci_env_chi32.json`) land in
`examples/out/` and are git-ignored.

### A note on JSON

Yastn's `to_dict` payloads embed numpy arrays and use integer dict keys for
`site_data` — neither survives a plain `json.dump` round-trip. The thin
`examples/_jsonio.py` helper adds a custom encoder + decoder + int-key fix
so the JSON files can be loaded back into yastn unchanged.

## Layout

```
.
├── README.md
├── yastn/                  # git submodule, the yastn package
└── examples/
    ├── CI_states/          # pre-converged CI state JSON + chi-{32,64,128} env JSON
    ├── _jsonio.py          # ndarray-aware JSON helper for yastn dicts
    ├── 00_yastn_basics.py  # primer: tensor init + contraction
    ├── 01_build_and_save.py
    ├── 02_load_and_ctm.py
    └── 03_measure.py
```
