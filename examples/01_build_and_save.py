"""
01 — Load the CI PEPS from CI_states/ and re-save it as JSON.

The CI states in ``examples/CI_states/`` were produced with a torch backend on a
``RectangularUnitcell`` lattice with pattern ``[[0,1],[1,0]]`` (Z2-symmetric
spinless fermions, D=4). The state JSON is in yastn's modern ``Peps.to_dict``
format (``type='Peps'``, ``dict_ver=1``), so ``yastn.from_dict`` can rebuild it
directly under our numpy backend; no pickle / no model wrapper needed.

This script reads that JSON, prints state info, and re-saves under a new
filename via :mod:`_jsonio` (round-trip sanity check + a stable input name
for scripts 02/03).
"""
import os, pickle

import yastn

import _jsonio

HERE = os.path.dirname(__file__)
CI_STATE_IN = os.path.join(HERE, "CI_states",
                           "Z2_t1_1.0_2x2_N2_D_4_chi_128_state.json")
OUT_DIR = os.path.join(HERE, "out")
STATE_PATH = os.path.join(OUT_DIR, "Z2_t1_1.0_2x2_N2_D_4_chi_128_state.json")


def load_CI_state(path=CI_STATE_IN, sym="Z2"):
    # initialize the yastn config
    config = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    # load state dict from json file
    with open(path, "r") as f:
        d = _jsonio.load(f)
    psi = yastn.from_dict(d, config=config)
    return psi


def save_state(psi, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        # `_jsonio.dump` uses the NumPy_Encoder convention from
        # tn-torch_dev_square so JSONs stay round-trip-compatible.
        _jsonio.dump(psi.to_dict(), f)
    print(f"saved state to {path}")


if __name__ == "__main__":
    psi = load_CI_state()
    print(f"geometry: {psi.geometry}")
    for site in psi.sites():
        print(f"site {site}: tensor shape = {psi[site].get_shape()}")
    save_state(psi, STATE_PATH)
