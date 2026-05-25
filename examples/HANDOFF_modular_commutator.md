# Handoff — continue the Rényi modular commutator experiment on a larger machine

Self-contained guide to resume on a cluster with more RAM. Full trap/decision
history: `DEVLOG_modular_commutator.md`. Physics: the project memory note
`renyi-modular-commutator.md`.

## TL;DR status
- **Goal:** extract the chiral central charge `c_-` of the stored Chern-insulator
  iPEPS via the Rényi modular commutator `ω_{n,n}` computed from the CTMRG env.
  Universal target for a Chern insulator: `c_- = 1`,
  `arg ω_{n,n} = -(π/12) q(n,n)`, `q(n,n)=2n²/((n+1)(2n+1))`
  (`arg ω_{1,1}=-π/36≈-0.0873`).
- **DONE & validated (2×2 window):** Strategy 1 (dense) == Strategy 2 (replica
  "virtual RDM") to 1e-18. Result `c_-(n=1)=-0.751`, χ-converged (32→64→128).
  Sign is convention-dependent (|c_-|≈0.75); the gap to 1 is window-size +
  finite-D (D=4 chiral-PEPS obstruction).
- **THE BLOCKER to fix on the cluster:** the general open-window RDM builder
  `rdm_window` is buggy (wrong multi-site fermionic signs). Fix it, then push
  window size to test whether |c_-| → 1.

## Files (all under `examples/`, import only `yastn` + these)
- `_modcomm.py` — everything. Strategy 1 (`omega_dense`, dense ref, VALIDATED),
  Strategy 2 (`omega_strategy2`/`_replica_contract`, VALIDATED, uses `rdm2x2`),
  the memory-safety layer (`_predict_peak_gb`, `_auto_slice`, `_contract_path`,
  `safe_rdm_window`), and the BUGGY general builder (`rdm_window`,
  `_open_double_layer`, `_rdm_window_tensor`).
- `_memguard.py` — `run_with_memory_cap` (forked child + psutil kill on low
  system-available RAM). Requires psutil.
- `05_modular_commutator.py` — driver (2×2 gate + χ table + von Neumann J).
- `DEVLOG_modular_commutator.md` — 15 documented traps + crash postmortem.
- Fixtures: `CI_states/Z2_..._chi_32_state_ctm_env.json` (committed). χ=64/128
  envs are git-ignored — regenerate from the state JSON (scripts 01/02 + CTMRG).

## Cluster setup
```bash
git submodule update --init --recursive          # yastn on branch oe_blocksparse (verify!)
#   git -C yastn branch --show-current   ->   oe_blocksparse
pip install -e yastn                              # or PYTHONPATH=$PWD/yastn
pip install psutil numpy scipy                    # psutil is REQUIRED by _memguard
cd examples
PYTHONPATH="$PWD/../yastn" python 05_modular_commutator.py   # reproduce 2×2
```
Use a Python with psutil + numpy. (On the dev Mac the trap was the conda *base*
env lacking psutil — see DEVLOG #9. On a cluster just ensure the env has psutil.)

## With more RAM, relax the safety knobs
- `_memguard.run_with_memory_cap(..., min_avail_gb=<~10% of RAM>, mem_gb=None)`.
  Keep `min_avail_gb` as the primary signal; leave `mem_gb=None` (per-process RSS
  over-counts COW fork pages — DEVLOG #13).
- `safe_rdm_window(env, H, W, budget_gb=<large>)` — predicts the peak and slices
  only if needed; with lots of RAM most windows won't need slicing.
- **Never call `yastn.ncon` on these networks** (default order OOMs, DEVLOG
  #11/#15). Use `_contract_path` (follows the memory-optimal path) or
  `safe_rdm_window`.

## STEP 1 (must do first): fix `rdm_window`
Diagnosis (confirmed): 2×2 single-site ⟨n⟩ from `rdm_window` matches `rdm2x2`
exactly, but the full RDM differs by ‖Δ‖=0.43 (NOT a bra/ket/conj convention).
`_open_double_layer` builds a position-INDEPENDENT tile; the correct RDM needs
POSITION-DEPENDENT physical-leg swap gates (cf. `rdm2x2`'s per-corner
`swap_gate(axes=(virtual,(s,s')))`, which differ for TL/TR/BL/BR — and a general
window also needs interior-site swaps).

Two fix routes:
1. Generalize `yastn/yastn/tn/fpeps/envs/rdm.py`'s `_append_vec_{tl,tr,bl,br}_open`
   + per-position swaps to an H×W tiling.
2. (Recommended) Adapt the ALREADY-CORRECT swap machinery in
   `yastn/yastn/tn/fpeps/envs/_env_ctm_measure.py::measure_nsite_exact`
   (`DoublePepsTensor.set_operator_` / `add_charge_swaps_`) to leave the physical
   legs OPEN instead of contracting an operator — that path handles general
   windows with correct fermionic strings.

**Validation (convention-free, do yastn-native, NOT dense numpy):** from the
window RDM compute ⟨n_i⟩, ⟨c†_i c_j⟩, ⟨n_i n_j⟩ and check against
`env.measure_1site` / `env.measure_nn` to ~1e-8. Also, at 2×2, the window RDM
must reduce (trace D=BL) to the same ρ_ABC that `_rho_abc_2x2` (rdm2x2-based)
gives, so `omega_strategy2`'s validated `Z(n=1)=0.07339758+0.00481921j` is
reproduced.

## STEP 2: window-size scaling (the actual experiment)
With a correct `rdm_window`:
- Four-region partition meeting at the central point (A,B,C window sectors + D
  = the rest; the CTM env is part of D). B is the SHARED region (in both ρ_AB and
  ρ_BC). Build ρ_ABC, ρ_AB=Tr_C, ρ_BC=Tr_A from the SAME window RDM (DEVLOG #6).
- The replica contraction is BOSONIC on the sign-resolved RDM legs (no extra
  swaps — DEVLOG, "Strategy 2"): reuse `_replica_contract`.
- Sweep windows 2×2 → 3×3 → 4×4 → … and report `c_- = extract_c_minus(ω_{1,1},1,1)`
  vs window size. Square windows preserve the 4-fold geometry; 4×4 = 16 sites
  (2^32 dense RDM) needs the boundary-MPS route or large RAM.
- For very large windows, the true memory-efficient method is a **boundary-MPS
  virtual contraction** (contract the replica network column-by-column with a
  truncated boundary bond dim) — the genuine "virtual RDM contract-physical-legs"
  approach. yastn's `EnvWindow` + `measure_nsite` (boundary-MPS) are the starting
  points.

## Expected outcome / open question
Does |c_-| → 1 as the window grows (benchmark reached), or stall (finite-D
obstruction)? 2×2 gives |c_-|≈0.75. This is THE open question the cluster run
should answer. Report c_-(n=1) and c_-(n=2) vs window size and χ.
