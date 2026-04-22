"""
02 — Load the saved fPEPS and converge its environment with CTMRG.

Uses `yastn.tn.fpeps.load_from_dict` to rebuild a `Peps` from JSON, then constructs
an `EnvCTM` and sweeps it to a fixed point with `env.ctmrg_`.

The final environment is serialized to JSON so script 03 can reuse it without
re-running CTMRG.
"""
import json
import os
import time

import yastn
import yastn.tn.fpeps as fpeps

HERE = os.path.dirname(__file__)
STATE_PATH = os.path.join(HERE, "out", "fci_square_state.json")
ENV_PATH = os.path.join(HERE, "out", "fci_square_env.json")


def load_state(path, sym="Z2"):
    config = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    with open(path, "r") as f:
        d = json.load(f)
    psi = fpeps.load_from_dict(config, d)
    return psi, config


def run_ctmrg(psi, config, chi=16, max_sweeps=50, corner_tol=1e-8):
    # Environment leg: boundary bond dimension chi, symmetry sectors mirroring the state.
    env_leg = yastn.Leg(config, s=1, t=(0, 1), D=(chi // 2, chi - chi // 2))
    env = fpeps.EnvCTM(psi, init="eye", leg=env_leg)

    t0 = time.perf_counter()
    info = env.ctmrg_(
        opts_svd={"D_total": chi, "tol": 1e-10, "truncate_multiplets": True},
        max_sweeps=max_sweeps,
        corner_tol=corner_tol,
    )
    print(f"CTMRG done in {time.perf_counter() - t0:.2f}s: {info}")
    return env


def save_env(env, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(env.save_to_dict(), f)
    print(f"saved env to {path}")


if __name__ == "__main__":
    psi, config = load_state(STATE_PATH)
    print(f"loaded state with {len(psi.sites())} unique sites")
    env = run_ctmrg(psi, config, chi=16, max_sweeps=50)
    save_env(env, ENV_PATH)
