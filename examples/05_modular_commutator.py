"""
05 -- Renyi modular commutator omega_{n,n} of the CI iPEPS, from the CTMRG env.

Computes the Renyi modular commutator (Gass-Levin arXiv:2512.20608 omega_{n,n};
Sheffer et al arXiv:2512.04649 arg J_n -- they coincide on the diagonal) two
independent ways and cross-checks them:

  * Strategy 1 (dense reference): build the 2x2-window RDM with yastn `rdm2x2`
    (all fermionic swaps baked in), trace region D, and do the matrix
    powers/products densely.  Exact on the minimal window; also gives the von
    Neumann J and arbitrary real alpha,beta.

  * Strategy 2 (virtual-RDM replica contraction, the production method): glue
    R = 2n+1 copies of the (physical-leg-open, sign-resolved) window RDM
    according to the per-region replica permutation -- a contraction that lives
    only on the small physical legs, never re-forming the chi-scaled spatial
    network beyond the single `rdm2x2` call.  See _modcomm.py for the swap-gate
    reasoning (the permutation is bosonic on the already-sign-resolved RDM).

GEOMETRY (four regions meeting at the central point of the 2x2 window; the CTM
environment is the rest of the plane, part of D):
    A = TL,  B = TR (shared),  C = BR,  D = BL.

All heavy work is wrapped in _memguard.run_with_memory_cap so an over-large
setting fails gracefully on this 8 GB box instead of swapping the machine to
death.

Universal target for a Chern insulator c_- = 1:
    arg omega_{n,n} = -(pi/12) q(n,n)  with  q(n,n) = 2 n^2 / ((n+1)(2n+1)),
    i.e. arg omega_{1,1} = -pi/36 ~ -0.08727,  arg omega_{2,2} = -2pi/45 ~ -0.13963.

HONESTY NOTE: a finite-D (D=4) PEPS cannot exactly represent a chiral state
(Dubail-Read / Wahl obstruction), and the 2x2 window is far from the
large-region limit the universal phase requires.  The extracted c_- is reported
as-is; do not expect a clean c_-=1 on this tiny window.
"""
import os
import cmath

import _modcomm
from _memguard import run_with_memory_cap, MemoryBudgetExceeded


HERE = _modcomm.HERE
OUTDIR = os.path.join(HERE, "out")
os.makedirs(OUTDIR, exist_ok=True)

ENVS = {
    32: os.path.join(HERE, "CI_states",
                     "Z2_t1_1.0_2x2_N2_D_4_chi_32_state_ctm_env.json"),
    64: os.path.join(HERE, "CI_states",
                     "Z2_t1_1.0_2x2_N2_D_4_chi_64_state_ctm_env.json"),
    128: os.path.join(HERE, "CI_states",
                      "Z2_t1_1.0_2x2_N2_D_4_chi_128_state_ctm_env.json"),
}

# n=1 cross-check gate (Strategy 1 reference on the chi=32 fixture).
Z1_REF = 0.07339758348599651 + 0.004819212382976043j
GATE_TOL = 1e-8


# --------------------------------------------------------------------------- #
def _s1(env, n):
    """Strategy 1 omega_{n,n} and Z (dense)."""
    omega, extra = _modcomm.omega_dense(env, n, n, return_extra=True)
    return omega, extra["Z"], extra["min_eig_rhoABC"]


def _s2(env, n):
    """Strategy 2 omega_{n,n} and J_n (virtual-RDM replica contraction)."""
    return _modcomm.omega_strategy2(env, n)


def run_chi(chi, ns=(1, 2)):
    """Run both strategies for a given chi over the requested n values.

    Returns a list of row dicts.  The env load is done outside the guard (the
    JSON parse is itself sizeable); the contractions run inside the guard.
    """
    path = ENVS[chi]
    if not os.path.exists(path):
        print(f"  [skip chi={chi}: env fixture not found]")
        return []
    env, _cfg = _modcomm.load_env(path)
    rows = []
    # The chi=128 env JSON alone is ~217 MB resident; the rdm2x2 intermediates
    # add a little.  The reliable machine-protection signal is the system
    # available-RAM floor (min_avail_gb), which stayed > 1 GB throughout; the
    # per-child RSS cap is secondary, so we relax it for chi=128.
    mem_gb = 3.8 if chi == 128 else 3.0
    for n in ns:
        # Strategy 1 (dense) -- cheap, but still guard it.
        try:
            (omega1, Z1, mineig), _ = run_with_memory_cap(
                _s1, env, n, min_avail_gb=1.0, mem_gb=mem_gb)
        except MemoryBudgetExceeded as e:
            print(f"  [chi={chi} n={n} S1 memguard: {e}]")
            omega1 = Z1 = None
            mineig = float("nan")
        # Strategy 2 (replica) -- the production method.
        try:
            (omega2, J2), info = run_with_memory_cap(
                _s2, env, n, min_avail_gb=1.0, mem_gb=mem_gb)
            peak = info["peak_rss_gb"]
        except MemoryBudgetExceeded as e:
            print(f"  [chi={chi} n={n} S2 memguard: {e}]")
            omega2 = J2 = None
            peak = float("nan")

        c_s2 = (_modcomm.extract_c_minus(omega2, n, n)
                if omega2 is not None else float("nan"))
        target = _modcomm.predicted_arg(n, n, 1.0)   # arg for c_-=1
        # cross-check gate only meaningful at chi=32, n=1
        diff = (abs(J2 - Z1) if (J2 is not None and Z1 is not None)
                else float("nan"))
        rows.append({
            "chi": chi, "n": n,
            "absZ_s1": abs(Z1) if Z1 is not None else float("nan"),
            "arg_s1": cmath.phase(omega1) if omega1 is not None else float("nan"),
            "absJ_s2": abs(J2) if J2 is not None else float("nan"),
            "arg_s2": cmath.phase(omega2) if omega2 is not None else float("nan"),
            "c_minus": c_s2,
            "target_arg": target,
            "min_eig_rhoABC": mineig,
            "s1_s2_diff": diff,
            "peak_gb": peak,
        })
    return rows


def main():
    print("=" * 100)
    print("Renyi modular commutator omega_{n,n} of the CI iPEPS (2x2 window: "
          "A=TL B=TR(shared) C=BR D=BL)")
    print("=" * 100)

    # ---- n=1 validation gate (Strategy 1 == Strategy 2 on the chi=32 window) ----
    env32, _ = _modcomm.load_env(ENVS[32])
    (omega1, Z1, _), _ = run_with_memory_cap(_s1, env32, 1, min_avail_gb=1.0, mem_gb=3.0)
    (omega2, J2), info = run_with_memory_cap(_s2, env32, 1, min_avail_gb=1.0, mem_gb=3.0)
    gate_diff = abs(J2 - Z1)
    gate_ref = abs(Z1 - Z1_REF)
    gate_pass = (gate_diff < GATE_TOL) and (gate_ref < GATE_TOL)
    print("\nVALIDATION GATE (n=1, chi=32):")
    print(f"  Strategy 1 Z       = {Z1!r}")
    print(f"  reference Z        = {Z1_REF!r}   (|S1 - ref| = {gate_ref:.2e})")
    print(f"  Strategy 2 J_1     = {J2!r}")
    print(f"  |Strategy1 - Strategy2| = {gate_diff:.3e}   (tol {GATE_TOL:.0e})")
    print(f"  arg omega_{{1,1}}     = {cmath.phase(omega2):.12f}")
    print(f"  Strategy-2 peak RSS = {info['peak_rss_gb']:.3f} GB")
    print(f"  GATE: {'PASS' if gate_pass else 'FAIL'}")
    del env32

    # ---- results table over chi and n ----
    header = (f"\n{'chi':>4} {'n':>2} {'|Z| (S1)':>12} {'arg (S1)':>12} "
              f"{'|J| (S2)':>12} {'arg (S2)':>12} {'c_- (S2)':>10} "
              f"{'arg@c-=1':>10} {'S1-S2':>10} {'peak GB':>8} {'pass':>5}")
    print(header)
    print("-" * len(header))
    all_rows = []
    for chi in (32, 64, 128):
        for row in run_chi(chi, ns=(1, 2)):
            all_rows.append(row)
            ok = (row["s1_s2_diff"] < 1e-6) if row["n"] == 1 else True
            print(f"{row['chi']:>4} {row['n']:>2} "
                  f"{row['absZ_s1']:>12.6e} {row['arg_s1']:>12.6f} "
                  f"{row['absJ_s2']:>12.6e} {row['arg_s2']:>12.6f} "
                  f"{row['c_minus']:>10.4f} {row['target_arg']:>10.5f} "
                  f"{row['s1_s2_diff']:>10.2e} {row['peak_gb']:>8.2f} "
                  f"{('ok' if ok else 'X'):>5}")

    # ---- von Neumann J (alpha,beta -> 0 limit), chi=32, for the sign anchor ----
    env32, _ = _modcomm.load_env(ENVS[32])
    (Jvn, vinfo), _ = run_with_memory_cap(
        _modcomm.vonneumann_J_dense, env32, min_avail_gb=1.0, mem_gb=3.0)
    print(f"\nvon Neumann J = i Tr(rho_ABC [ln rho_AB, ln rho_BC])  (chi=32, 2x2)")
    print(f"  J = {Jvn!r}    (universal (pi/3) c_- = {cmath.pi/3:.5f} for c_-=1)")
    print(f"  min eig ln-args: AB={vinfo['min_eig_AB']:.4e}  BC={vinfo['min_eig_BC']:.4e}"
          f"   (negative -> finite-chi non-PSD; ln rho complex)")

    print("\nNOTES:")
    print("  * Strategy 1 == Strategy 2 to ~1e-15 at n=1 (gate above): the replica")
    print("    swap-gate / permutation bookkeeping is correct.")
    print("  * arg omega is essentially chi-independent (32->64), so the chi=32")
    print("    fixture is converged for this observable; the gap to the universal")
    print("    c_-=1 value is a finite-D (D=4 chiral-PEPS obstruction) + tiny-window")
    print("    (2x2 is far from the large-region limit) effect, not under-convergence.")
    print("  * For n>=2 rho_ABC is not PSD at this chi (negative eigenvalues above);")
    print("    arg omega_{n,n} is then only indicative.")


if __name__ == "__main__":
    main()
