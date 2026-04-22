"""
01 — Build a small checkerboard spinless-fermion fPEPS and save it to JSON.

This shows the minimum recipe: pick a lattice geometry, pick fermionic operators
with a chosen abelian symmetry, build rank-5 on-site tensors, wrap them in a
`Peps`, and serialize via `Peps.to_dict`.
"""
import json
import os

import yastn
import yastn.tn.fpeps as fpeps

OUT_DIR = os.path.join(os.path.dirname(__file__), "out")
STATE_PATH = os.path.join(OUT_DIR, "fci_square_state.json")


def build_state(D=2, sym="Z2", seed=0):
    # yastn config: fermionic abelian tensors with numpy backend.
    config = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    config.backend.random_seed(seed)

    # Spinless fermion operators (c, c^+, n, identity) with matching symmetry.
    ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend,
                                           default_dtype="complex128")

    # 2-site checkerboard unit cell (A / B sublattices), matching the FCI_square setup.
    geometry = fpeps.CheckerboardLattice()

    # Virtual leg shared across the four auxiliary directions of each on-site tensor.
    # For Z2 we split D between the two charge sectors.
    if sym == "Z2":
        t_sectors = (0, 1)
        D_sectors = (D // 2, D - D // 2)
    else:
        t_sectors = (0, 1)
        D_sectors = (D // 2, D - D // 2)
    vleg = yastn.Leg(config, s=1, t=t_sectors, D=D_sectors)
    pleg = ops.space()  # physical leg carried by the fermion operators

    # Rank-5 on-site tensor: legs [top, left, bottom, right, physical].
    def random_site():
        return yastn.rand(
            config,
            legs=[vleg.conj(), vleg, vleg, vleg.conj(), pleg],
        )

    psi = fpeps.Peps(geometry, tensors={s: random_site() for s in geometry.sites()})
    return psi


def save_state(psi, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(psi.to_dict(), f)
    print(f"saved state to {path}")


if __name__ == "__main__":
    psi = build_state(D=2, sym="Z2")
    for site in psi.sites():
        print(f"site {site}: tensor shape = {psi[site].get_shape()}")
    save_state(psi, STATE_PATH)
