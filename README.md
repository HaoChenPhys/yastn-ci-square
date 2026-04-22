# yastn FCI-square example

Minimal, self-contained demo of using [yastn](https://github.com/yastn/yastn)
directly for a fermionic iPEPS calculation on the square lattice (the setting
behind the `FCI_square` example). Only `yastn` is imported — no wrapper code.

## What it shows

1. **Build and save a state** — construct a small Z2-symmetric spinless-fermion
   iPEPS on a checkerboard unit cell, serialize it with `Peps.to_dict` + `json`.
2. **Load and converge an environment** — rebuild the state with
   `fpeps.load_from_dict`, instantiate an `EnvCTM`, and sweep it to a fixed
   point with `env.ctmrg_`.
3. **Measure observables** — use `env.measure_1site` / `env.measure_nn` to get
   site occupations, nearest-neighbour hoppings, density-density correlators,
   and the energy of a spinless t-V Hamiltonian.

## Setup

```bash
git clone --recurse-submodules <this-repo-url>
cd yastn-fci-square-example

# If you already cloned without --recurse-submodules:
git submodule update --init --recursive

pip install -e yastn            # install the bundled yastn submodule
```

`yastn` is pinned as a git submodule, so the example always runs against a
known-good revision.

## Run

```bash
cd examples
python 01_build_and_save.py
python 02_load_and_ctm.py
python 03_measure.py
```

Artifacts (`fci_square_state.json`, `fci_square_env.json`) land in
`examples/out/` and are git-ignored.

## Layout

```
.
├── README.md
├── yastn/                  # git submodule, the yastn package
└── examples/
    ├── 01_build_and_save.py
    ├── 02_load_and_ctm.py
    └── 03_measure.py
```
