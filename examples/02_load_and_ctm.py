"""
02 — Load the CI-state CTMRG environment at chi=32 and re-save it as JSON.

The chi=32 JSON in ``CI_states/`` already holds a fully converged ``EnvCTM``
(``psi``/``env``/``proj``), so no CTMRG sweep is needed here. We just rebuild
the environment with ``yastn.from_dict`` under our numpy config and serialize
it back out to a local path for script 03. A round-trip check at the end
confirms the saved JSON reloads to a bit-identical env (matching ``<n_i>``).
"""
import os

import yastn
import _jsonio

HERE = os.path.dirname(__file__)
CI_ENV_IN = os.path.join(HERE, "CI_states",
                         "Z2_t1_1.0_2x2_N2_D_4_chi_32_state_ctm_env.json")
ENV_PATH = os.path.join(HERE, "out", "ci_env_chi.json")


def load_CI_env(path=CI_ENV_IN, sym="Z2"):
    # Same numpy override pattern as script 01 — overrides whatever
    # backend/device was stored in the file.
    config = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    with open(path, "r") as f:
        d = _jsonio.load(f)
    env = yastn.from_dict(d, config=config)
    return env, config


def load_env_from_json(path=ENV_PATH, sym="Z2"):
    config = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    with open(path, "r") as f:
        d = _jsonio.load(f)
    return yastn.from_dict(d, config=config)


def save_env(env, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        # ndarray-aware encoder; see _jsonio.py for the round-trip rules.
        _jsonio.dump(env.to_dict(), f)
    print(f"saved env to {path}")


if __name__ == "__main__":
    env, _ = load_CI_env()
    # env.psi is a Peps2Layers (bra ⊗ ket); the ket is the physical state.
    psi = env.psi.ket if hasattr(env.psi, "ket") else env.psi
    print(f"loaded env with {len(psi.sites())} unique sites "
          f"(geometry: {type(psi.geometry).__name__})")
    # Each EnvCTM site stores 8 boundary tensors:
    # 4 corners (tl/tr/bl/br) and 4 edges (t/b/l/r).
    sample = env[next(iter(psi.sites()))]
    print(f"env tensor shapes at {next(iter(psi.sites()))}:")
    for dirn in sample.fields():
        print(f"  {dirn}: {getattr(sample, dirn).get_shape()}")
    save_env(env, ENV_PATH)

    # --- Round-trip check: reload the JSON we just wrote and verify that a
    #     cheap observable matches the value measured on the original env.
    env2 = load_env_from_json(ENV_PATH)
    ops = yastn.operators.SpinlessFermions(sym="Z2", backend=env.psi.config.backend,
                                           default_dtype="complex128")
    occ = env.measure_1site(ops.n())
    occ2 = env2.measure_1site(ops.n())
    max_diff = max(abs(occ[s] - occ2[s]) for s in occ)
    print(f"\nJSON round-trip check for saved env:")
    for s in occ:
        print(f"  <n> at {s}: orig={occ[s].real:+.10f}  reloaded={occ2[s].real:+.10f}")
    print(f"  max |Δ<n>| = {max_diff:.2e}")
