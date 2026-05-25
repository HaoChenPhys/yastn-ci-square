# Devlog — Rényi modular commutator experiment

Running log of what was built and, importantly, the **traps hit and how to
avoid them**. Plan: `~/.claude/plans/sharded-kindling-lake.md`. Physics:
memory `renyi-modular-commutator.md`.

## Components
- `_modcomm.py` — Strategy 1 (dense exact reference, validated) + Strategy 2
  (virtual-RDM replica contraction, in progress).
- `_memguard.py` — `run_with_memory_cap(fn, ..., min_avail_gb, mem_gb)`: runs a
  heavy contraction in a forked child and kills it before RAM starves. **Use it
  for every heavy contraction.**
- `05_modular_commutator.py` — driver (in progress).

## Validated facts (anchors)
- `rdm2x2(s0)` returns a normalized 8-leg RDM, leg order
  `[TL,TL', BL,BL', TR,TR', BR,BR']` (interleaved bra/ket); sites
  `TL=s0, BL=nn'b', TR=nn'r', BR=nn'br'`. Trace = 1.
- Four-region geometry: A=TL, B=TR (shared), C=BR, D=BL; env = rest = part of
  `(ABC)^c`. `ρ_ABC = Tr_D ρ_W`.
- **Validation gate (Strategy 2 must reproduce):** n=1 at the 2×2 window,
  `Z = Tr(ρ_ABC ρ_AB ρ_BC) = 0.07339758348599651 + 0.004819212382976043j`,
  `arg(ω_{1,1}) = 0.065564895138`. (n=1 needs NO matrix powers.)
- n=1 extracted `c_- ≈ -0.75` (sane for a tiny region; |c_-|=1 reduced by size,
  sign is convention-dependent).

## TRAPS & FIXES (don't repeat these)

1. **Finite-χ CTM RDMs are NOT positive-semidefinite.** ρ_W(2×2) at χ=32 has
   eigenvalues down to ≈ −0.10 (real, sum to 1, Hermitian — just not PSD). The
   2-site ρ_AB/ρ_BC happened to be PSD, but ρ_ABC is not.
   - ⇒ Integer powers ρ^n are fine; **real powers / log(ρ) are ill-defined**
     (von Neumann J came out ≈0+complex garbage). Higher n amplifies the
     negative-eigenvalue error (n=2,3 extracted c_- blew up to −2.8, −18).
   - ⇒ Prefer **n=1 (no powers)** for the cleanest signal; treat real-α ω and
     von Neumann J as best-effort only, and report the negativity.

2. **macOS memory accounting is treacherous.**
   - `resource.setrlimit(RLIMIT_AS, ...)` is REJECTED ("current limit exceeds
     maximum limit") even with hard=∞ — RLIMIT_AS is effectively unsupported.
     Don't rely on it.
   - `ps -o rss` (and psutil rss) **undercount** badly — macOS compresses idle
     pages (saw 4 GB allocation report ~0.5 GB RSS).
   - `ps -o vsz` / psutil vms **overcount** uselessly (~394 GB reserved space).
   - ⇒ The reliable kill signal is **system available memory**
     (`psutil.virtual_memory().available`); kill the child before it hits a
     floor (~1 GB). This box is **8 GB total, ~2 GB free** — slicing is
     mandatory, not optional.

3. **Don't test a memory guard with constant-value arrays.** `np.ones(...)+1`
   is all-identical → macOS compresses 16 GB to ~nothing → guard never trips
   (false negative). Use `np.random.rand(...)` (incompressible) — then the
   guard correctly fires at the available-RAM floor.

4. **Fermionic Hermitian-conjugate check is misleading.** Naive
   `ρ.conj().transpose(swap bra/ket)` gives `||ρ-ρ^H|| ≈ 0.5` even for a valid
   RDM — fermionic conj needs swap gates. **Validate Hermiticity/spectrum by
   fusing to a matrix and `eigh`** (real eigenvalues ⇒ Hermitian), not by a
   naive transpose.

5. **Dense (Strategy 1) only works up to ~2×2.** `to_dense` of a 2N-leg RDM is
   `2^{2N}`. 2×2 (8 legs) = fine; 3×3 (18 legs) = 256 GB. Large windows REQUIRE
   the virtual-RDM contraction (contract physical legs; never materialize the
   dense physical RDM).

6. **yastn rdm builders use differing internal sign conventions.** My
   partial-traced ρ_BC ≠ `rdm2x1` output (but same spectrum) — a convention
   difference, not a bug. ⇒ Derive ρ_ABC, ρ_AB, ρ_BC all from the **same** ρ_W
   so they share one convention; don't mix builders in one product.

7. **Always run from `examples/`** with `PYTHONPATH="$PWD/../yastn"` (scripts
   use bare `import _jsonio` / `import _modcomm` and `__file__`-relative paths).
   Only the χ=32 env JSON is committed; χ=64/128 must be regenerated via CTMRG.

## Progress log

### Strategy 2 implemented & VALIDATED (2×2)
- `_modcomm.py` Strategy-2 (`omega_strategy2`, `_replica_contract`, `Jn_strategy2`)
  + driver `05_modular_commutator.py`. All heavy work under `_memguard`.
- **Validation gate PASS:** Strategy 1 == Strategy 2 to **8.7e-19** at n=1 and
  1.3e-18 at n=2; |S1(n=1) − reference Z| = 0 (exact).
- **Key resolution of "Risk #1" (swap gates):** the replica permutation acts
  *bosonically* on the already-sign-resolved RDM physical legs — **no extra swap
  gates at replica crossings**. All fermionic signs are inside `rdm2x2`'s
  double-layer construction (same reason Strategy 1's dense matrix powers carry
  no signs). Confirmed by the 1e-18 agreement.
- **Result (2×2, four regions):** c_- = −0.751 (n=1), −2.837 (n=2), vs target
  +1. arg ω₁₁ is χ-converged (0.065565→0.065530→0.065523 for χ=32→64→128), so the
  gap is NOT under-convergence in χ. Sign is convention-dependent (|c_-|≈0.75).
  von Neumann J ≈ 0 (vs π/3) — the 2×2 window is far too small for the universal
  value. Peak RSS: χ=32 ~0.3–0.5 GB, χ=64 ~1.4 GB, χ=128 ~3.5 GB (RDM route).

### Trap 8 — two routes, opposite memory scaling (crossover)
- "RDM route" (form the open physical RDM via `rdm2x2`, then bosonic replica
  contraction): cheap at small windows, but the open RDM is `2^{2N}` (area) →
  blows up for large windows. yastn only ships an open-RDM builder up to 2×2.
- "Flat/virtual route" (ket/bra separate, contract physical legs early via the
  replica network): memory ~ `D^{2·perimeter}·χ` (perimeter) → ~3 GB even at
  2×2 (small-N regime where it LOSES), but the right choice for LARGE windows.
- ⇒ The user's "virtual RDM saves memory" is true in the large-region limit
  (where we must go for the universal value), false at 2×2. Crossover is at
  larger N. Scaling needs the virtual route + aggressive `unroll` slicing, OR a
  general open-RDM builder for moderate windows (≤ ~3×4 = 12 sites on 8 GB).

### NEXT (decisive): window-size scaling
2×2 alone cannot distinguish "window too small" from "finite-D obstruction".
Push to 2×3, 3×3 (and 3×4 if RAM allows) with a four-region partition around the
centre, and report c_-(n=1) vs window size: does |c_-| → 1 (benchmark
approaching) or stall (finite-D)? This is the open task.

## MEMORY-CRASH POSTMORTEM (the machine was killed twice) + more traps

Root causes of the repeated overflows, and the fixes now in place:

9. **Wrong interpreter → guard silently absent.** `which python3` is the conda
   *base* env, which has NO psutil; only `torch_peps` does. `_memguard` imports
   psutil, so under the wrong python it failed to import and the guard was
   bypassed. FIX: always run with the explicit interpreter
   `/opt/homebrew/Caskroom/miniforge/base/envs/torch_peps/bin/python3`; the
   guard now raises a LOUD error if psutil is missing. The Bash tool does NOT
   inherit an interactive `conda activate`, so pin the path.

10. **Reactive RSS polling cannot stop a single big allocation.** A multi-GB
    `numpy`/yastn allocation commits faster than the 0.05 s poll and freezes the
    Mac before the kill fires. FIX: **proactive prediction** — `_modcomm`
    `_predict_peak_gb` / `_auto_slice` / `safe_rdm_window` call
    `get_contraction_path` (cheap, shapes only) to bound `largest_intermediate`
    BEFORE allocating, slice the heavy bonds to fit a budget, or REFUSE.

11. **`yastn.ncon` uses the DEFAULT (by-label) order — it blows up.** The 2×3
    open-window RDM hit 2.3 GB via ncon vs 0.2 GB on the memory-optimal path.
    And `contract_with_unroll` only accepts an ALL-pairwise path
    (`oe_blocksparse.py:813` needs every step len==2), which the optimal path
    violates (it has len-1 steps; the ">2 tensors" error message is misleading).
    FIX: `_contract_path` executes the optimal path step-by-step with
    `yastn.tensordot` (bounded peak == prediction; handles len-1 steps; carries
    fermionic signs). `rdm_window` now uses it for the no-slice case.

12. **`rdm_window` (the general open-window builder, agent-written) is BUGGY —
    confirmed.** `omega_strategy2`'s 1e-18 validation uses the `rdm2x2`-based
    `_rho_abc_2x2`, NOT `rdm_window`. Careful 2×2 comparison vs the trusted
    `rdm2x2`:
      * single-site ⟨n⟩ MATCHES exactly [0.501,0.499,0.499,0.501] (an earlier
        "0.5 error" was a bug in my dense marginal-check script, not the builder);
      * but the FULL RDM differs by ‖Δ‖=0.43 (‖ref‖=0.51), and it is NOT a
        bra↔ket / conjugation convention (tested Aref, conj, pair-transpose,
        Hermitian-conj — all ≈0.43).
    ⇒ the *multi-site fermionic signs* are wrong (occupations are insensitive to
    them). `_open_double_layer` builds a POSITION-INDEPENDENT tile, but the
    correct RDM needs POSITION-DEPENDENT physical-leg swap gates (cf. rdm2x2's
    per-corner `swap_gate(axes=(virtual,(s,s')))` lines, which differ for
    TL/TR/BL/BR — and a general window also needs interior-site swaps rdm2x2
    never had to handle). FIX = generalize rdm2x2's per-position swap logic to
    H×W, OR adapt the validated swap machinery in
    `_env_ctm_measure.measure_nsite_exact` (`DoublePepsTensor.add_charge_swaps_`)
    to leave physical legs open. Validate by ⟨c†c⟩/⟨nn⟩ vs `env.measure_nn`
    (convention-free), done yastn-native (not dense numpy). NEEDS a higher-RAM
    machine to iterate safely (see below).

15. **`ncon` OOMs even UNDER the guard.** A guarded 2×2 `ncon` (default order)
    was SIGKILLed (exitcode -9) by the OS while `min_avail` still read 1.09 GB —
    the allocation spiked between 0.03 s polls. ⇒ reactive guarding is not
    enough on its own; NEVER call `ncon` here. Only the proactively-predicted
    `_contract_path` (optimal order) is safe. Even just *building* the two 2×2
    RDM tensors for comparison peaks ~1.3 GB (env+fork baseline), so debugging
    the builder fix on this 8 GB box is not safely possible.

13. **COW fork inflates child RSS.** A forked child inherits the parent's loaded
    env (~1.2 GB) as copy-on-write; `ps`/psutil RSS counts those shared pages,
    so a tiny contraction shows ~1.2 GB RSS. ⇒ the per-child `mem_gb` RSS cap
    false-trips; **rely on `min_avail_gb` (system available) as the primary kill
    signal** and set `mem_gb` high or None.

14. **NEVER run unguarded on this box.** An "innocent" 2×2 diagnostic that
    called `ncon` (bad order) unguarded got OOM-killed (exit 137). Every run —
    even a quick check — must go through `_memguard.run_with_memory_cap`.

### STATUS / honest conclusion
- **Solid & validated:** modular commutator from CTM at the 2×2 window;
  Strategy 1 == Strategy 2 to 1e-18; c_-(n=1) = -0.751, χ-converged (32→128).
  Memory safety is now robust (proactive prediction + hardened guard prevented
  further crashes — children were killed gracefully, machine survived).
- **Blocked:** reaching the universal value needs LARGE regions. Clean square
  scaling jumps 2×2 → 4×4 (2^32 RDM, impossible). Non-square windows need the
  general `rdm_window`, which is (a) buggy (#12) and (b) memory-blocked on this
  8 GB box (~1–2.6 GB free, shared with other apps; env+fork baseline ~1.2 GB).
  `contract_with_unroll` slicing is also blocked by the pairwise-path limit
  (#11). So the benchmark c_-→1 is NOT reachable here without: fixing the
  `rdm_window` builder + a boundary-MPS (true virtual-RDM) contraction + more
  free RAM (close other apps / bigger machine). The 2×2 |c_-|≈0.75 + finite-D
  chiral-PEPS obstruction is the documented limitation (plan stop-point ii).
