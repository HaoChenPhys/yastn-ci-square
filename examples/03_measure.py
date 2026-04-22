"""
03 — Measure observables from the converged CTM environment.

Loads state + environment saved by scripts 01 and 02 and evaluates:
  * site occupations  <n_i>
  * nearest-neighbour hopping  <c^+_i c_j>
  * nearest-neighbour density-density  <n_i n_j>
  * energy per site of the chiral t-V checkerboard Hamiltonian used to
    optimize the CI states (see ``models/fermion/square_tv.py`` in
    ``tn-torch_dev_square``):

        H =  -t1 sum_<ij>_h e^{+i*phi*sgn} c^+_i c_j  +  h.c.    (NN, h-bonds)
            -t1 sum_<ij>_v e^{-i*phi*sgn} c^+_i c_j  +  h.c.    (NN, v-bonds)
            -t2 sum_<ik>   c^+_i c_k     +  h.c.                (NNN, sub-lat sign)
            -t3 sum_<ill'> c^+_i I c_l'  +  h.c.                (3rd NN, straight line)
            +V1 sum_<ij>   n_i n_j  +  V2 sum_<ik> n_i n_k  -  mu sum_i n_i

    For the bundled CI state the canonical parameter choice is
    ``t1=1, t2=t1/(2+sqrt(2)), t3=t1/(2+2*sqrt(2)), phi=pi/4`` (V1=V2=mu=0),
    giving the reference energy ``E/site = -1.0217145...``.

All correlators use ``EnvCTM.measure_1site``, ``measure_nn``,
``measure_2x2``, and ``measure_line``. No machinery outside yastn is needed.
"""
import os
import math

import yastn
from yastn.tn.fpeps import Bond

import _jsonio

HERE = os.path.dirname(__file__)
STATE_PATH = os.path.join(HERE, "out", "Z2_t1_1.0_2x2_N2_D_4_chi_128_state.json")
ENV_PATH = os.path.join(HERE, "out", "ci_env_chi.json")


def load_state_and_env(sym="Z2"):
    # Same Z2 fermionic / numpy / complex128 config used to write the JSONs.
    # Passing it to from_dict overrides any backend stored in the file.
    config = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    with open(STATE_PATH, "r") as f:
        psi = yastn.from_dict(_jsonio.load(f), config=config)
    with open(ENV_PATH, "r") as f:
        env = yastn.from_dict(_jsonio.load(f), config=config)
    # Operators must share the same symmetry / backend as the state so the
    # block structure matches when measure_* contract them.
    ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend,
                                           default_dtype="complex128")
    return psi, env, ops


def ci_default_params(t1=1.0):
    """Canonical parameter choice for the bundled CI states.

    Mirrors ``CI_square.py:117`` in tn-torch_dev_square, where t2/t3/phi are
    derived from t1.
    """
    return dict(t1=t1,
                t2=t1 / (2 + math.sqrt(2)),
                t3=t1 / (2 + 2 * math.sqrt(2)),
                phi=math.pi / 4,
                V1=0.0, V2=0.0, mu=0.0)


def energy_per_site(psi, env, ops, *, t1, t2, t3, phi, V1=0.0, V2=0.0, mu=0.0):
    """Port of ``tV_checkerboard_model.energy_per_site`` (kinetic + V + mu).

    The model is checkerboard-chiral: NN hopping carries a phase ``e^{±i*phi}``
    whose sign flips between sublattices; vertical bonds use the conjugate of
    the horizontal phase; NNN hopping ``t2`` flips sign between sublattices;
    3rd-NN ``t3`` is a straight-line ``c^+ I c`` along x and y.
    """
    n, c, cp, I = ops.n(), ops.c(), ops.cp(), ops.I()
    nsites = len(psi.sites())

    e_on = e_h = e_v = e_diag = e_adiag = e_3nn = 0

    for site in psi.sites():
        # Sub-lattice A/B sign for the chiral phase (and t2 sign).
        if (site[0] + site[1]) % 2 == 0:
            t1_eff = t1 * complex(math.cos(phi), math.sin(phi))    # t1 * e^{+i*phi}
            t2_eff = t2
        else:
            t1_eff = t1 * complex(math.cos(phi), -math.sin(phi))   # t1 * e^{-i*phi}
            t2_eff = -t2

        # Chemical potential
        e_on += -mu * env.measure_1site(n, site=site)

        # Horizontal NN bond (right neighbour)
        h_bond = Bond(site, psi.nn_site(site, "r"))
        loc = V1 * env.measure_nn(n, n, bond=h_bond)
        res = -t1_eff * env.measure_nn(cp, c, bond=h_bond)
        loc += (res + res.conjugate()).real
        e_h += loc

        # Vertical NN bond (bottom neighbour) — uses conj(t1_eff) by convention.
        v_bond = Bond(site, psi.nn_site(site, "b"))
        loc = V1 * env.measure_nn(n, n, bond=v_bond)
        res = -t1_eff.conjugate() * env.measure_nn(cp, c, bond=v_bond)
        loc += (res + res.conjugate()).real
        e_v += loc

        # NNN diagonal (br) — anchored at `site`.
        site_br = psi.nn_site(site, "br")
        loc = V2 * env.measure_2x2(n, n, sites=[site, site_br])
        res = -t2_eff * env.measure_2x2(cp, c, sites=[site, site_br])
        loc += (res + res.conjugate()).real
        e_diag += loc

        # NNN anti-diagonal: cp at `site`, c at south-west neighbour.
        site_bl = psi.nn_site(site, (1, -1))
        loc = V2 * env.measure_2x2(n, n, sites=[site, site_bl])
        res = t2_eff * env.measure_2x2(cp, c, sites=[site, site_bl])
        loc += (res + res.conjugate()).real
        e_adiag += loc

        # 3rd-NN: straight line c^+ I c along x and y.
        site_r, site_r2 = psi.nn_site(site, (0, 1)), psi.nn_site(site, (0, 2))
        site_b, site_b2 = psi.nn_site(site, (1, 0)), psi.nn_site(site, (2, 0))
        res = -t3 * env.measure_line(cp, I, c, sites=[site, site_r, site_r2])
        e_3nn += (res + res.conjugate()).real
        res = -t3 * env.measure_line(cp, I, c, sites=[site, site_b, site_b2])
        e_3nn += (res + res.conjugate()).real

    return ((e_on + e_h + e_v + e_diag + e_adiag + e_3nn) / nsites).real


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

    params = ci_default_params(t1=1.0)
    e = energy_per_site(psi, env, ops, **params)
    print(f"\nEnergy per site (CI Hamiltonian, "
          f"t1={params['t1']}, t2={params['t2']:.6f}, t3={params['t3']:.6f}, "
          f"phi=pi/4, V1=V2=mu=0):")
    print(f"  E/site = {e:+.10f}")
    print(f"  reference (chi=128, fullrank): -1.0217145251")
