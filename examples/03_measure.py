"""
03 — Measure observables from the converged CTM environment.

Loads state + environment saved by scripts 01 and 02 and evaluates:
  * site occupations  <n_i>
  * nearest-neighbour hopping  <c^+_i c_j>  (and hermitian conjugate)
  * nearest-neighbour density-density  <n_i n_j>
  * energy per site of a spinless t-V Hamiltonian,
        H = -t sum_<ij> (c^+_i c_j + h.c.) + V sum_<ij> n_i n_j

All correlators use `EnvCTM.measure_1site` and `EnvCTM.measure_nn`, which return
dicts keyed by site / bond. No machinery outside yastn is needed.
"""
import json
import os

import yastn
import yastn.tn.fpeps as fpeps

HERE = os.path.dirname(__file__)
STATE_PATH = os.path.join(HERE, "out", "fci_square_state.json")
ENV_PATH = os.path.join(HERE, "out", "fci_square_env.json")


def load_state_and_env(sym="Z2"):
    config = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    with open(STATE_PATH, "r") as f:
        psi = fpeps.load_from_dict(config, json.load(f))
    with open(ENV_PATH, "r") as f:
        env = fpeps.load_from_dict(config, json.load(f))
    ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend,
                                           default_dtype="complex128")
    return psi, env, ops


def energy_per_site(env, ops, t=1.0, V=0.0):
    cp_c = env.measure_nn(ops.cp(), ops.c())      # <c^+_i c_j>
    c_cp = env.measure_nn(ops.c(), ops.cp())      # <c_i c^+_j>  (= -<c^+_j c_i> up to sign)
    nn = env.measure_nn(ops.n(), ops.n())         # <n_i n_j>

    # CheckerboardLattice has 2 unique sites and 4 nn bonds per 2-site unit cell.
    nsites = len(env.sites())
    nbonds = len(cp_c)

    hop = sum(cp_c.values()) - sum(c_cp.values())  # kinetic: -t * (c^+c + h.c.) summed over bonds
    inter = sum(nn.values())

    # Per-site normalization: Nbonds / Nsites bonds per site (=2 for the square lattice).
    return (-t * hop + V * inter).real / nsites


if __name__ == "__main__":
    psi, env, ops = load_state_and_env()

    occ = env.measure_1site(ops.n())
    print("Site occupations <n_i>:")
    for site, val in occ.items():
        print(f"  {site}: {val.real:+.6f}")

    hop = env.measure_nn(ops.cp(), ops.c())
    print("\nNearest-neighbour hopping <c^+_i c_j>:")
    for bond, val in hop.items():
        print(f"  {bond}: {val:+.6f}")

    nn = env.measure_nn(ops.n(), ops.n())
    print("\nNearest-neighbour density-density <n_i n_j>:")
    for bond, val in nn.items():
        print(f"  {bond}: {val.real:+.6f}")

    e = energy_per_site(env, ops, t=1.0, V=0.0)
    print(f"\nEnergy per site of H = -t sum_<ij>(c^+c + h.c.) with t=1, V=0: {e:+.6f}")
