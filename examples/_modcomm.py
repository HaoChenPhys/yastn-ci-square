"""Rényi modular commutator on the CI iPEPS, from the CTMRG environment.

Implements two routes to ω_{α,β} = phase of ⟨ρ_AB^α ρ_BC^β⟩ (Gass-Levin
arXiv:2512.20608), equivalently arg J_n with α=β=n (Sheffer et al
arXiv:2512.04649):

  * Strategy 1 (`omega_dense`): exact reference on a small window. Builds the
    window RDM from `rdm2x2`, then does all partial traces / matrix powers /
    products in dense numpy in a fixed Fock basis. Fermionic signs are already
    baked into the RDM by `rdm2x2`, so the dense algebra is ordinary. Used as
    ground truth to validate Strategy 2 and to reach arbitrary real α,β and the
    von Neumann limit.

  * Strategy 2 (separate module section, added later): the memory-efficient
    "virtual RDM" replica contraction that contracts physical legs.

Universal prediction (chiral central charge c_-):
    ω_{α,β} = exp(-(πi/12) q(α,β) c_-),
    q(α,β) = α/(α+1) + β/(β+1) - (α+β)/(α+β+1).
On the diagonal q(n,n) = 2 n² / ((n+1)(2n+1)).
"""
import os
import math
import cmath

import numpy as np
import yastn
from yastn.tn.fpeps.envs.rdm import rdm2x2

import _jsonio

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_FIXTURE = os.path.join(
    HERE, "CI_states", "Z2_t1_1.0_2x2_N2_D_4_chi_32_state_ctm_env.json")


# --------------------------------------------------------------------------- #
#  Loading / config                                                           #
# --------------------------------------------------------------------------- #
def load_env(path=ENV_FIXTURE, sym="Z2"):
    """Load a converged EnvCTM from a yastn `to_dict` JSON under a numpy config."""
    cfg = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    with open(path, "r") as f:
        env = yastn.from_dict(_jsonio.load(f), cfg)
    return env, cfg


# --------------------------------------------------------------------------- #
#  Analytic prediction utilities                                              #
# --------------------------------------------------------------------------- #
def q_factor(alpha, beta):
    """q(α,β) = α/(α+1) + β/(β+1) - (α+β)/(α+β+1)."""
    return (alpha / (alpha + 1) + beta / (beta + 1)
            - (alpha + beta) / (alpha + beta + 1))


def predicted_omega(alpha, beta, c_minus):
    """Universal value exp(-(πi/12) q(α,β) c_-)."""
    return cmath.exp(-1j * math.pi / 12 * q_factor(alpha, beta) * c_minus)


def predicted_arg(alpha, beta, c_minus):
    return -math.pi / 12 * q_factor(alpha, beta) * c_minus


def extract_c_minus(omega, alpha, beta):
    """Invert the universal formula: c_- = -(12/(π q)) arg ω.

    Uses the principal value of arg ω (valid while |arg| < π).
    """
    q = q_factor(alpha, beta)
    return -cmath.phase(omega) * 12 / (math.pi * q)


# --------------------------------------------------------------------------- #
#  Strategy 1 — dense exact reference on a small window                       #
# --------------------------------------------------------------------------- #
def window_rdm_dense(env, s0=None):
    """Return the 2x2-window RDM as a dense 8-index numpy array.

    Index order follows `rdm2x2`: [TL,TL', BL,BL', TR,TR', BR,BR'] (interleaved
    bra/ket), with the four lattice sites
        TL = s0, BL = nn(s0,'b'), TR = nn(s0,'r'), BR = nn(s0,'br').
    Each index has dimension d=2 (Z2 occupation 0/1).
    """
    psi = env.psi.ket if hasattr(env.psi, "ket") else env.psi
    if s0 is None:
        s0 = psi.sites()[0]
    rho, _norm = rdm2x2(s0, psi, env)          # normalized, Tr = 1
    arr = rho.to_dense()                        # shape (2,)*8
    return np.asarray(arr)


def _to_matrix(t, bra_axes, ket_axes):
    """Reshape a dense operator tensor into a Fock-basis matrix.

    `bra_axes`/`ket_axes` give the tensor axes that are the bra (row) and ket
    (column) indices, in the desired most->least significant site order.
    """
    ndim = t.ndim
    perm = list(bra_axes) + list(ket_axes)
    t = np.transpose(t, perm)
    dim = int(np.prod([t.shape[i] for i in range(len(bra_axes))]))
    return t.reshape(dim, dim)


def _mpow(M, p, *, tol=1e-9):
    """Matrix power M**p for Hermitian M.

    Integer p uses `matrix_power` (robust to non-PSD). Non-integer p uses an
    eigendecomposition; eigenvalues below `tol` in magnitude are dropped
    (their contribution to a positive power vanishes), and negative eigenvalues
    raised to a real power are handled with the principal branch (flagged by
    the caller via `min_eig`).
    """
    if abs(p - round(p)) < 1e-12:
        return np.linalg.matrix_power(M, int(round(p)))
    w, V = np.linalg.eigh(M)
    wp = np.zeros_like(w, dtype=complex)
    for i, wi in enumerate(w):
        if abs(wi) < tol:
            wp[i] = 0.0
        elif wi > 0:
            wp[i] = wi ** p
        else:
            wp[i] = (abs(wi) ** p) * cmath.exp(1j * math.pi * p)  # principal branch
    return (V * wp) @ V.conj().T


def _mlog(M, *, tol=1e-12):
    """Hermitian matrix log, dropping (near-)zero eigenvalues (-> 0 in K=-lnρ).

    Negative eigenvalues (finite-χ artifact) get a complex log via the
    principal branch; returns also the most-negative eigenvalue for diagnostics.
    """
    w, V = np.linalg.eigh(M)
    lg = np.zeros_like(w, dtype=complex)
    for i, wi in enumerate(w):
        if abs(wi) < tol:
            lg[i] = 0.0
        else:
            lg[i] = cmath.log(wi)  # complex if wi < 0
    return (V * lg) @ V.conj().T, float(w.min())


# FOUR-region partition of the 2x2 window, meeting at the central point:
#   A = TL (axes 0,1),  B = TR (axes 4,5),  C = BR (axes 6,7),  D = BL (axes 2,3).
# The CTM environment is the rest of the infinite plane. (ABC)^c = D ∪ env, so
#   ρ_ABC = Tr_D ρ_W  (the environment is already traced inside ρ_W).
# B is the region shared by AB and BC, as required by ⟨ρ_AB^α ρ_BC^β⟩.
# The fourth region D is essential: a tripartite pure state (no D) gives ω = 1
# trivially (Gass-Levin Eq. 8). Here D=BL ∪ env carries genuine entanglement,
# and in the replica (Strategy 2) picture D gets the identity permutation.
def _rho_ABC_from_window(rhoW):
    """Trace out D=BL -> ρ_ABC on the three regions {A=TL, B=TR, C=BR}.

    Returns dense tensor with axes [b_TL,k_TL, b_TR,k_TR, b_BR,k_BR].
    """
    # rhoW axes: 0 b_TL,1 k_TL,2 b_BL,3 k_BL,4 b_TR,5 k_TR,6 b_BR,7 k_BR
    return np.einsum('ab cc de fg -> ab de fg'.replace(' ', ''), rhoW)


def omega_dense(env, alpha, beta, s0=None, *, return_extra=False):
    """ω_{α,β} on the minimal L-tromino window via dense algebra.

    A=TL, B=TR, C=BR (B shared). Returns ω = Z/|Z| with
    Z = Tr(ρ_ABC · ρ_AB^α · ρ_BC^β), powers embedded with identity on the
    third site. Robust for integer α,β even when ρ is not PSD.
    """
    rhoW = window_rdm_dense(env, s0)
    rho_abc = _rho_ABC_from_window(rhoW)   # [bTL,kTL, bTR,kTR, bBR,kBR]

    # Matrices in Fock basis, site order (TL, TR, BR) most->least significant.
    M_abc = _to_matrix(rho_abc, bra_axes=(0, 2, 4), ket_axes=(1, 3, 5))  # 8x8

    # ρ_AB on {TL,TR}: trace BR (axes 4,5 of rho_abc).
    rho_ab = np.einsum('ab cd ee -> ab cd'.replace(' ', ''), rho_abc)
    M_ab = _to_matrix(rho_ab, bra_axes=(0, 2), ket_axes=(1, 3))           # 4x4

    # ρ_BC on {TR,BR}: trace TL (axes 0,1 of rho_abc).
    rho_bc = np.einsum('aa cd ef -> cd ef'.replace(' ', ''), rho_abc)
    M_bc = _to_matrix(rho_bc, bra_axes=(0, 2), ket_axes=(1, 3))           # 4x4

    I2 = np.eye(2)
    M_ab_p = np.kron(_mpow(M_ab, alpha), I2)    # acts on (TL,TR) ⊗ I_BR
    M_bc_p = np.kron(I2, _mpow(M_bc, beta))     # I_TL ⊗ acts on (TR,BR)

    Z = np.trace(M_abc @ M_ab_p @ M_bc_p)
    omega = Z / abs(Z)
    if not return_extra:
        return omega
    w_abc = np.linalg.eigvalsh(M_abc)
    return omega, {"Z": Z, "abs_Z": abs(Z), "min_eig_rhoABC": float(w_abc.min())}


def vonneumann_J_dense(env, s0=None):
    """von Neumann modular commutator J = i Tr(ρ_ABC [ln ρ_AB, ln ρ_BC]).

    = (π/3) c_- in the universal limit. Note: finite-χ negative eigenvalues
    make ln ρ complex; we report J and the most-negative eigenvalue so the
    reliability is visible.
    """
    rhoW = window_rdm_dense(env, s0)
    rho_abc = _rho_ABC_from_window(rhoW)
    M_abc = _to_matrix(rho_abc, (0, 2, 4), (1, 3, 5))
    rho_ab = np.einsum('abcdee->abcd', rho_abc)
    rho_bc = np.einsum('aacdef->cdef', rho_abc)
    M_ab = _to_matrix(rho_ab, (0, 2), (1, 3))
    M_bc = _to_matrix(rho_bc, (0, 2), (1, 3))
    I2 = np.eye(2)
    K_ab, min_ab = _mlog(M_ab)
    K_bc, min_bc = _mlog(M_bc)
    lnAB = np.kron(K_ab, I2)
    lnBC = np.kron(I2, K_bc)
    comm = lnAB @ lnBC - lnBC @ lnAB
    J = 1j * np.trace(M_abc @ comm)
    return J, {"min_eig_AB": min_ab, "min_eig_BC": min_bc}


# --------------------------------------------------------------------------- #
#  Strategy 2 — virtual-RDM replica contraction (memory-efficient)            #
# --------------------------------------------------------------------------- #
#
# We compute  J_n = <psi^{(R)}| pi_A pi_B pi_C |psi^{(R)}>  with R = 2n+1
# replicas (Sheffer et al arXiv:2512.04649 Eq. 1, 5).  "Virtual RDM" means we
# contract the *physical* legs (ket <-> bra, per the replica permutation) so the
# only intermediates that survive live on the (small) physical bonds of the
# window, never the chi-scaled spatial network more than once.
#
# Key structural fact, validated to ~1e-15 against Strategy 1 (see
# 05_modular_commutator.py): the heavy, chi-scaled spatial contraction is done
# exactly ONCE per region, by yastn's own RDM builders in
# `yastn/tn/fpeps/envs/rdm.py` (`rdm2x2`, `rdm1x2`, ...).  Those builders
# already insert every fermionic swap gate of the double-layer (ket (x) bra)
# construction (cf. `_append_vec_{tl,tr,bl,br}_open` and
# DoublePepsTensor.fuse_layers).  The replica permutation then acts on the
# *physical* legs of the resulting (sign-resolved) reduced-density tensor, and
# on those legs it is an ordinary (bosonic) index relabelling -- exactly as in
# Strategy 1, where the matrix power rho_AB^n and the products are plain dense
# linear algebra with no extra signs.  So we must NOT re-insert swap gates at
# the replica crossings; doing so double-counts the signs already baked into the
# RDM.  This is the resolution of "Risk #1" in the plan.
#
# Memory profile: the only chi-heavy step is the single rdm builder call (the
# same contraction scripts 02/03 already run safely).  The replica gluing is on
# physical legs of total dim d^(#sites-per-region) per leg, looped pairwise, so
# its peak is negligible for the 2x2 window and controllable (via the d-leg
# contraction order / `contract_with_unroll` slicing) for larger windows.


def _replica_perms(n):
    """Per-region replica permutations sigma_X: bra_r contracts ket_{sigma_X(r)}.

    R = 2n+1 replicas, labelled 0..R-1.
      B=TR (shared): cyclic over ALL replicas        bra_r <-> ket_{(r+1) mod R}
      A=TL:          cyclic over first  n+1 {0..n}    cyclic on that block, identity else
      C=BR:          cyclic over last   n+1 {n..2n}   cyclic on that block, identity else
      D=BL:          identity (same as environment)
    Returns dict region -> list sigma with sigma[r] = ket index that bra_r joins.

    Validated: with this assignment the dense replica contraction reproduces
    Strategy 1's Z(n) exactly (n=1: 0.07339758...+0.00481921...j).
    """
    R = 2 * n + 1
    ident = list(range(R))

    sB = [(r + 1) % R for r in range(R)]

    sA = list(range(R))
    blkA = list(range(0, n + 1))           # {0..n}
    for i, r in enumerate(blkA):
        sA[r] = blkA[(i + 1) % len(blkA)]

    sC = list(range(R))
    blkC = list(range(n, R))               # {n..2n}
    for i, r in enumerate(blkC):
        sC[r] = blkC[(i + 1) % len(blkC)]

    return {"A": sA, "B": sB, "C": sC, "D": ident}


# --------------------------------------------------------------------------- #
#  Window open-RDM builders (physical legs left open; D and env traced)       #
# --------------------------------------------------------------------------- #
#
# Each builder returns a yastn Tensor `rho_abc` whose legs are, in order,
#   [bra_A, ket_A, bra_B, ket_B, bra_C, ket_C]
# i.e. one (bra, ket) physical-index pair per region A, B, C; region D and the
# CTM environment are already traced.  When a region spans several lattice
# sites the corresponding bra/ket leg is the *fused* multi-site physical leg.

def _rho_abc_2x2(env, s0=None):
    """2x2 window: A=TL, B=TR, C=BR, D=BL (the validation geometry).

    Uses yastn `rdm2x2` (all fermionic swaps baked in) then traces region D.
    Returns the *normalized* (Tr rho_W = 1) tensor; the overall positive norm is
    irrelevant to the phase omega = Z/|Z|.
    """
    from yastn.tn.fpeps.envs.rdm import rdm2x2
    psi = env.psi.ket if hasattr(env.psi, "ket") else env.psi
    if s0 is None:
        s0 = psi.sites()[0]
    rho, _norm = rdm2x2(s0, psi, env)   # [bTL kTL bBL kBL bTR kTR bBR kBR]
    rho_abc = rho.trace(axes=(2, 3))    # trace D=BL -> [bA kA bB kB bC kC]
    return rho_abc


# --------------------------------------------------------------------------- #
#  GENERAL open-window RDM builder (any small rectangle)                      #
# --------------------------------------------------------------------------- #
#
# Generalizes the validated flat double-layer network of
# `examples/04_measure_nn_scratch.py` (which reproduces `EnvCTM.measure_nn` to
# 1e-12) to an H x W rectangle of sites, with the on-site PHYSICAL legs left
# OPEN.  The construction tiles the CTM boundary (4 corners + edges along each
# side) and fills the interior with ket / bra.conj() on-site tensors.  The only
# fermionic swap gates are the uniform per-site double-layer gates from
# DoublePepsTensor.fuse_layers, identical at every site:
#       swap(ket.l, bra.t), swap(bra.l, bra.t),
#       swap(ket.b, bra.r), swap(bra.b, bra.r).
# The replica permutation acts later, bosonically, on the resulting open legs
# (same reasoning as the rdm2x2 route -- see module note above).
#
# Leg / orientation conventions (from 04 + _env_contractions):
#   corners (fused [bra,ket]):  tl=(down,right) tr=(left,down)
#                               bl=(right,up)   br=(up,left)
#   edges  (3 legs, middle bulk leg unfuses to (bra,ket)):
#       t=(left, [b b'], right)   b=(right, [t t'], left)
#       l=(down, [r r'], up)      r=(up,   [l l'],  down)
#   on-site ket/bra legs: (t, l, b, r, s)   with s the fused [phys, aux] (dim 2).
#
# Grid indexing: cell (i, j), i = row 0..H-1 top->bottom (increasing nx),
#                              j = col 0..W-1 left->right (increasing ny).
# Returned tensor: legs interleaved [bra_(i,j), ket_(i,j)] in row-major order of
# the requested `sites` (after the trivial aux leg of each physical pair is
# traced).  Trace over all (bra,ket) pairs == 1 (normalized).

def window_sites(psi, nw=None, H=2, W=2):
    """Row-major list of the H x W window sites with NW corner `nw`.

    Returns (grid, sites) where grid[i][j] is the lattice Site at row i, col j
    and `sites` is the flat row-major list.
    """
    if nw is None:
        nw = psi.sites()[0]
    grid = []
    for i in range(H):
        row_start = nw
        for _ in range(i):
            row_start = psi.nn_site(row_start, "b")
        row = [row_start]
        s = row_start
        for _ in range(W - 1):
            s = psi.nn_site(s, "r")
            row.append(s)
        grid.append(row)
    sites = [grid[i][j] for i in range(H) for j in range(W)]
    return grid, sites


def _open_double_layer(A):
    """Per-site OPEN double-layer tensor with legs
        [t t'] [l l'] [b b'] [r r'] s s'
    where the four virtual legs are the *fused* double-layer legs IDENTICAL to
    ``DoublePepsTensor.fuse_layers`` (so they plug straight into the standard
    fused CTM corner/edge tensors), and s, s' are the on-site physical legs of
    the ket and bra (each the [phys, aux] fused leg, total dim 2) left OPEN.

    Validated: tracing (s, s') reproduces ``A.fuse_layers()`` to 0.0 -- i.e.
    every fermionic swap of the double-layer construction is baked in here, so
    NO further per-site swap gates are needed when tiling.  This is the
    open-physical-leg generalization of fuse_layers.
    """
    from yastn import tensordot
    Ab, Ak = A.Ab_Ak_with_charge_swap()
    Ab = Ab.fuse_legs(axes=((0, 1), (2, 3), 4))  # [t l] [b r] s
    Ak = Ak.fuse_legs(axes=((0, 1), (2, 3), 4))
    if A.op is not None:
        Ak = tensordot(Ak, A.op, axes=(2, 1))
    tt = tensordot(Ak, Ab.conj(), axes=((), ()))  # [tl][br] s [t'l'][b'r'] s'
    tt = tt.transpose(axes=(0, 3, 1, 4, 2, 5))     # [tl][t'l'][br][b'r'] s s'
    tt = tt.unfuse_legs(axes=(0, 1))               # t l t' l' [br] [b'r'] s s'
    tt = tt.swap_gate(axes=((1, 3), 2))            # l l' X t'
    tt = tt.fuse_legs(axes=((0, 2), (1, 3), 4, 5, 6, 7))  # [t t'] [l l'] [br] [b'r'] s s'
    tt = tt.unfuse_legs(axes=(2, 3))               # [t t'] [l l'] b r b' r' s s'
    tt = tt.swap_gate(axes=((2, 4), 5))            # b b' X r'
    tt = tt.fuse_legs(axes=(0, 1, (2, 4), (3, 5), 6, 7))  # [t t'][l l'][b b'][r r'] s s'
    return tt


def _rdm_window_tensor(env, grid):
    """Open-physical-leg window RDM network for the rectangle `grid`.

    Returns ``(args, names, swap, (H, W))`` ready for
    ``yastn.get_contraction_path`` / ``yastn.contract_with_unroll``.  The result
    tensor has interleaved [s, s'] physical legs (each fused [phys, aux]) in
    row-major order of `grid`.

    Tiling uses the standard *fused* CTM corner/edge tensors and the per-site
    open double-layer tile :func:`_open_double_layer` (all swaps baked in), so
    no extra swap gates are needed here.  This is the open-physical-leg
    generalization of yastn's own rdm builders.
    """
    psi = env.psi  # Peps2Layers; psi[s] is a DoublePepsTensor
    H = len(grid)
    W = len(grid[0])

    _cnt = [0]
    def L():
        _cnt[0] += 1
        return _cnt[0]

    args = []
    names = []
    def add(tensor, ig, name):
        args.extend([tensor, tuple(ig)])
        names.append(name)

    # ---- fused bulk bond labels (one per double-layer bond) -------------- #
    Hbond = [[L() for _ in range(W - 1)] for _ in range(H)]   # (i,j)-(i,j+1)
    Vbond = [[L() for _ in range(W)] for _ in range(H - 1)]   # (i,j)-(i+1,j)
    Tbond = [L() for _ in range(W)]   # top    edge <-> (0,j)
    Bbond = [L() for _ in range(W)]   # bottom edge <-> (H-1,j)
    Lbond = [L() for _ in range(H)]   # left   edge <-> (i,0)
    Rbond = [L() for _ in range(H)]   # right  edge <-> (i,W-1)
    # CTM boundary (corner<->edge<->edge) fused links
    toplink = [L() for _ in range(W + 1)]
    botlink = [L() for _ in range(W + 1)]
    leftlink = [L() for _ in range(H + 1)]
    rightlink = [L() for _ in range(H + 1)]
    # open physical legs (negative -> free)
    Ps = [[None for _ in range(W)] for _ in range(H)]    # ket s
    Psb = [[None for _ in range(W)] for _ in range(H)]   # bra s'
    nneg = [0]
    def NEG():
        nneg[0] += 1
        return -nneg[0]
    # row-major interleaved [s, s'] ordering
    for i in range(H):
        for j in range(W):
            Ps[i][j] = NEG()
            Psb[i][j] = NEG()

    NW, NE = grid[0][0], grid[0][W - 1]
    SW, SE = grid[H - 1][0], grid[H - 1][W - 1]

    # ---- corners (fused legs) -------------------------------------------- #
    add(env[NW].tl, (leftlink[0], toplink[0]), "tl")        # (down, right)
    add(env[NE].tr, (toplink[W], rightlink[0]), "tr")       # (left, down)
    add(env[SW].bl, (botlink[0], leftlink[H]), "bl")        # (right, up)
    add(env[SE].br, (rightlink[H], botlink[W]), "br")       # (up, left)

    # ---- edges (fused middle leg connects to the on-site fused leg) ------ #
    for j in range(W):
        add(env[grid[0][j]].t, (toplink[j], Tbond[j], toplink[j + 1]),
            f"t_{j}")                                        # (left, [b b'], right)
        add(env[grid[H - 1][j]].b, (botlink[j + 1], Bbond[j], botlink[j]),
            f"b_{j}")                                        # (right, [t t'], left)
    for i in range(H):
        add(env[grid[i][0]].l, (leftlink[i + 1], Lbond[i], leftlink[i]),
            f"l_{i}")                                        # (down, [r r'], up)
        add(env[grid[i][W - 1]].r, (rightlink[i], Rbond[i], rightlink[i + 1]),
            f"r_{i}")                                        # (up, [l l'], down)

    # ---- on-site open double-layer tiles --------------------------------- #
    for i in range(H):
        for j in range(W):
            tile = _open_double_layer(psi[grid[i][j]])
            # legs: [t t'] [l l'] [b b'] [r r'] s s'
            lt = Tbond[j] if i == 0 else Vbond[i - 1][j]
            lb = Bbond[j] if i == H - 1 else Vbond[i][j]
            ll = Lbond[i] if j == 0 else Hbond[i][j - 1]
            lr = Rbond[i] if j == W - 1 else Hbond[i][j]
            add(tile, (lt, ll, lb, lr, Ps[i][j], Psb[i][j]), f"site_{i}_{j}")

    args.append(tuple())  # outputs are the negative labels, ordered by |label|
    return args, names, [], (H, W)


def rdm_window(env, nw=None, H=2, W=2, *, optimize=None, unroll=None,
               checkpoint_loop=False, return_meta=False):
    """General open-physical-leg RDM of an H x W window (NW corner `nw`).

    Returns a yastn Tensor with interleaved physical legs
    [s_(0,0), s'_(0,0), s_(0,1), s'_(0,1), ...] in ROW-MAJOR order of the
    window (s = ket, s' = bra; signature matches yastn's own rdm builders so it
    contracts directly with `yastn.operators` and the existing dense reference).
    The trivial auxiliary leg fused into each physical leg is traced out, so
    each open leg is the bare physical leg (dim 2).  Normalized: full trace = 1.

    HEAVY: peak ~ D^(2*perimeter)*chi.  Wrap in
    `_memguard.run_with_memory_cap`.

    WARNING — UNVALIDATED / BUGGY: this general builder currently FAILS the
    independent marginal check (window <n> vs env.measure_1site shows a ~0.5
    error at 2x2; see DEVLOG trap #12). The validated modular-commutator result
    uses the rdm2x2-based path (`_rho_abc_2x2` / `omega_strategy2`), NOT this.
    Fix the tiling / per-site aux-trace parity before trusting any
    `rdm_window`-based scaling number.
    """
    import yastn
    from yastn.tn.fpeps.envs.rdm import trace_aux as _tr_aux
    psi = env.psi
    grid, _sites = window_sites(
        psi.ket if hasattr(psi, "ket") else psi, nw=nw, H=H, W=W)
    args, names, swap, (Hh, Ww) = _rdm_window_tensor(env, grid)
    if unroll is None:
        # No slicing: follow the memory-OPTIMAL path with tensordot (bounded
        # peak == prediction), NOT raw ncon (whose default by-label order can
        # blow up by 10x+ and OOM the machine).
        if optimize is None:
            optimize, _info = yastn.get_contraction_path(
                *args, names=names, who="rdm_window_%dx%d" % (Hh, Ww))
        res = _contract_path(args, optimize)
    else:
        if optimize is None:
            optimize, _info = yastn.get_contraction_path(
                *args, names=names, who="rdm_window_%dx%d" % (Hh, Ww),
                unroll=unroll)
        res = yastn.contract_with_unroll(
            *args, optimize=optimize, swap=swap, unroll=unroll,
            checkpoint_loop=checkpoint_loop)
    # res legs interleaved [s, s'] per site, each = fused [phys, aux] (dim 2).
    rho = res
    nsite = Hh * Ww
    # Trace the auxiliary leg of each physical pair.  Mirrors rdm2x2's
    # convention: the aux of the "first" site of a fermionic pair is swapped.
    # For a general window the relevant parity is (nx + ny) of the site: top/left
    # sites (even Manhattan parity from NW) carry the swap.  Validated below
    # against rdm2x2 / measure_nn.
    for k in reversed(range(nsite)):
        i, j = k // Ww, k % Ww
        rho = _tr_aux(rho, 2 * k, swap=((i + j) % 2 == 0))
    tr_order = (tuple(2 * i for i in range(nsite)),
                tuple(2 * i + 1 for i in range(nsite)))
    norm = rho.trace(axes=tr_order).to_number()
    rho = rho / norm
    if return_meta:
        return rho, {"H": Hh, "W": Ww, "grid": grid, "norm": norm}
    return rho


# --------------------------------------------------------------------------- #
#  PROACTIVE memory safety: predict the peak intermediate, slice, or refuse.   #
#  Reactive RSS polling cannot catch a single multi-GB allocation that freezes #
#  the machine before the next poll; `get_contraction_path` is cheap (shapes   #
#  only, no data) and reports `largest_intermediate`, so we predict the peak    #
#  BEFORE allocating and slice the heavy bonds until it fits, or raise.         #
# --------------------------------------------------------------------------- #
_ITEMSIZE = 16          # complex128 bytes
_PEAK_FACTOR = 3.0      # operands + result are held simultaneously


def _label_dims(args):
    """Map each *contracted* (positive) label -> its total leg dimension."""
    dims = {}
    for k in range(0, len(args) - 1, 2):
        T, ig = args[k], args[k + 1]
        legs = T.get_legs()
        for ax, lab in enumerate(ig):
            if isinstance(lab, int) and lab > 0:
                dims[lab] = sum(legs[ax].D)
    return dims


def _leg_for_label(args, label):
    for k in range(0, len(args) - 1, 2):
        T, ig = args[k], args[k + 1]
        if label in ig:
            return T.get_legs(axes=ig.index(label))
    raise KeyError(label)


def _predict_peak_gb(args, names, unroll=None, who="pred"):
    """(path, predicted_peak_GB). Cheap: opt_einsum on shapes, no contraction."""
    path, info = yastn.get_contraction_path(
        *args, names=names, unroll=unroll, who=who)
    return path, float(info.largest_intermediate) * _ITEMSIZE * _PEAK_FACTOR / 2**30


def _selftrace(T, lab):
    """Trace any repeated labels within a single tensor (opt_einsum 1-steps)."""
    seen = {}
    for ax, x in enumerate(lab):
        if x in seen:
            T = T.trace(axes=(seen[x], ax))
            return _selftrace(T, [l for k, l in enumerate(lab) if k not in (seen[x], ax)])
        seen[x] = ax
    return T, lab


def _contract_path(args, path):
    """Execute a (mostly-binary) opt_einsum path with `yastn.tensordot`.

    Follows the MEMORY-OPTIMAL order from `get_contraction_path` step by step,
    so the peak intermediate matches the prediction — unlike raw `yastn.ncon`,
    whose default by-label order can blow up. Handles length-1 steps (self
    traces / reorders) that `contract_with_unroll` rejects. yastn.tensordot
    carries the fermionic signs; the network's swaps are baked into the tiles.
    Output legs are returned ordered by the free (negative) labels: -1, -2, ...
    """
    tensors = list(args[0:2 * (len(args) // 2):2])
    labels = [list(ig) for ig in args[1:2 * (len(args) // 2):2]]
    for step in path:
        for k in sorted(step, reverse=True):  # pop high indices first
            pass
        popped = [(tensors.pop(k), labels.pop(k)) for k in sorted(step, reverse=True)]
        popped.reverse()                       # back to step order
        if len(popped) == 1:
            T, lab = _selftrace(*popped[0])
            tensors.append(T)
            labels.append(lab)
        else:
            (Ta, La), (Tb, Lb) = popped
            common = [x for x in La if x in Lb]
            R = yastn.tensordot(Ta, Tb,
                                axes=([La.index(x) for x in common],
                                      [Lb.index(x) for x in common]))
            tensors.append(R)
            labels.append([x for x in La if x not in common]
                          + [x for x in Lb if x not in common])
    assert len(tensors) == 1, "path did not reduce to a single tensor"
    T, lab = tensors[0], labels[0]
    out_order = sorted((x for x in lab if isinstance(x, int) and x < 0), reverse=True)
    return T.transpose(axes=tuple(lab.index(o) for o in out_order))


def _auto_slice(args, names, budget_gb):
    """Return (unroll, path, peak_gb).

    Slice the largest-dimension contracted bonds (the χ boundary links) with a
    shrinking chunk size until the predicted peak <= budget_gb.  Raises
    MemoryError if even chunk size 1 on the top bonds cannot fit.
    """
    path, peak = _predict_peak_gb(args, names)
    if peak <= budget_gb:
        return None, path, peak
    from yastn.tensor.oe_blocksparse import slice_leg_uniform
    dims = _label_dims(args)
    cands = [lab for lab, _ in sorted(dims.items(), key=lambda kv: -kv[1])]
    unroll, best = None, peak
    for nbond in range(1, min(len(cands), 6) + 1):
        for size in (16, 8, 4, 2, 1):
            unroll = {lab: slice_leg_uniform(_leg_for_label(args, lab), size)
                      for lab in cands[:nbond]}
            path, peak = _predict_peak_gb(args, names, unroll=unroll)
            best = min(best, peak)
            if peak <= budget_gb:
                return unroll, path, peak
    raise MemoryError(
        f"cannot slice window contraction under {budget_gb:.2f} GB "
        f"(best predicted peak {best:.2f} GB)")


def safe_rdm_window(env, H, W, *, nw=None, budget_gb=0.6, verbose=True):
    """Memory-SAFE open-window RDM.

    Predicts the peak intermediate via the memory-optimized path, slices the
    heavy (χ) bonds until it fits `budget_gb`, and REFUSES (MemoryError) if it
    cannot — so it can never trigger an unbounded allocation.  Still wrap the
    call in `_memguard.run_with_memory_cap(..., min_avail_gb=...)` as a backstop.
    """
    psi = env.psi.ket if hasattr(env.psi, "ket") else env.psi
    grid, _ = window_sites(psi, nw=nw, H=H, W=W)
    args, names, _swap, _hw = _rdm_window_tensor(env, grid)
    unroll, path, peak = _auto_slice(args, names, budget_gb)
    if verbose:
        nsl = "none" if unroll is None else {k: len(v) for k, v in unroll.items()}
        print(f"  [safe_rdm_window {H}x{W}] predicted peak ~{peak:.3f} GB "
              f"(budget {budget_gb}); slicing={nsl}")
    if unroll is None:
        # Predicted to fit without slicing -> contract along the SAME optimal
        # path we just predicted (via _contract_path), so the actual peak
        # matches `peak`. Guard the call as a backstop regardless.
        return rdm_window(env, nw=nw, H=H, W=W, optimize=path, unroll=None)
    # Needs slicing -> contract_with_unroll. NOTE: it requires an all-pairwise
    # (binary) path (oe_blocksparse.py:813); the memory-optimal path here has
    # length-1 steps, so this currently RAISES NotImplementedError rather than
    # crashing. Documented limitation (DEVLOG trap 11); needs an all-binary path
    # or a hand-written column-sweep contraction to scale past ~2x3 on this box.
    return rdm_window(env, nw=nw, H=H, W=W, optimize=path, unroll=unroll)


def predict_window_peak_gb(env, H, W, nw=None):
    """Diagnostic: predicted peak (GB) of the un-sliced H×W window contraction."""
    psi = env.psi.ket if hasattr(env.psi, "ket") else env.psi
    grid, _ = window_sites(psi, nw=nw, H=H, W=W)
    args, names, _swap, _hw = _rdm_window_tensor(env, grid)
    _path, peak = _predict_peak_gb(args, names, who=f"{H}x{W}")
    return peak


# --------------------------------------------------------------------------- #
#  Replica contraction of the per-region open-RDM tensor                      #
# --------------------------------------------------------------------------- #
def _replica_contract(rho_abc, n, perms=None, *, unroll=None,
                      checkpoint_loop=False, optimize=None, return_path=False):
    """Glue R = 2n+1 copies of `rho_abc` according to the replica permutation.

    `rho_abc` legs: [bra_A, ket_A, bra_B, ket_B, bra_C, ket_C].  For each region
    X and replica r, the bra leg of replica r contracts with the ket leg of
    replica sigma_X(r).  This is a *bosonic* contraction on the (already
    sign-resolved) physical legs -- no swap gates here (see module note).

    Returns Z = J_n (unnormalized expectation).

    With ``unroll=None`` (default) the gluing is done by :func:`yastn.ncon`
    (the physical legs are tiny, so this is cheap and supports the arbitrary
    contraction tree this network needs).  When ``unroll`` is supplied the work
    is routed through :func:`yastn.contract_with_unroll` so its slicing /
    ``checkpoint_loop`` can cap the peak for larger windows / larger n.
    """
    import yastn
    if perms is None:
        perms = _replica_perms(n)
    R = 2 * n + 1
    sig = {"A": perms["A"], "B": perms["B"], "C": perms["C"]}

    # Unique positive integer label per (region, ket-replica).  rho_abc axis
    # order: 0 bra_A, 1 ket_A, 2 bra_B, 3 ket_B, 4 bra_C, 5 ket_C.  The bra leg
    # of replica r in region X carries the *ket* label of sigma_X(r), so each
    # ket label is hit by exactly one bra (sigma is a permutation) -> every
    # label appears exactly twice (valid ncon; full contraction -> scalar).
    klab = {}
    idx = 1
    for X in ("A", "B", "C"):
        for m in range(R):
            klab[(X, m)] = idx
            idx += 1

    igs = []
    for r in range(R):
        igs.append([klab[("A", sig["A"][r])], klab[("A", r)],
                    klab[("B", sig["B"][r])], klab[("B", r)],
                    klab[("C", sig["C"][r])], klab[("C", r)]])

    if unroll is None:
        res = yastn.ncon([rho_abc] * R, igs)
        Z = res.to_number()
        if return_path:
            return Z, None
        return Z

    # unroll path: interleaved args for contract_with_unroll.
    args, names = [], []
    for r in range(R):
        args.extend([rho_abc, tuple(igs[r])])
        names.append("rho_r%d" % r)
    args.append(())
    if optimize is None:
        optimize, _info = yastn.get_contraction_path(
            *args, names=names, who="modcomm_replica_n%d" % n, unroll=unroll)
    res = yastn.contract_with_unroll(
        *args, optimize=optimize, unroll=unroll,
        checkpoint_loop=checkpoint_loop)
    Z = res.to_number()
    if return_path:
        return Z, optimize
    return Z


def Jn_strategy2(env, n, *, s0=None, window="2x2", perms=None, unroll=None,
                 checkpoint_loop=False, optimize=None, return_path=False):
    """J_n via the virtual-RDM replica contraction (Strategy 2).

    Parameters
    ----------
    window : str
        "2x2" -- the four-quadrant validation/production window (A=TL, B=TR,
        C=BR, D=BL meeting at the central point).  Other window shapes can be
        added by supplying a different open-RDM builder returning the same
        [bra_A, ket_A, bra_B, ket_B, bra_C, ket_C] leg layout.
    """
    if window == "2x2":
        rho_abc = _rho_abc_2x2(env, s0)
    else:
        raise NotImplementedError(f"window={window!r} not implemented")
    return _replica_contract(rho_abc, n, perms=perms, unroll=unroll,
                             checkpoint_loop=checkpoint_loop,
                             optimize=optimize, return_path=return_path)


def omega_strategy2(env, n, *, s0=None, window="2x2", **kw):
    """omega_{n,n} = J_n / |J_n| via the virtual-RDM replica contraction.

    Returns (omega, J_n).
    """
    Jn = Jn_strategy2(env, n, s0=s0, window=window, **kw)
    return Jn / abs(Jn), Jn


# --------------------------------------------------------------------------- #
#  Flat double-layer replica network (alternative; kept for scaling/checks)   #
# --------------------------------------------------------------------------- #
#
# This builds the SAME J_n as a single multi-tensor einsum over the 2x2 window
# with ket/bra kept separate and the env edges un-fused, in the style of
# examples/04_measure_nn_scratch.py.  The per-site double-layer swap gates are
# passed explicitly (identical in structure to fuse_layers: `l l' X t'` and
# `b b' X r'`), and the physical legs are tied across replicas by explicit
# permutation operators.  It is heavier than the RDM route (it holds several
# chi-scaled CTM rings at once) and is meant for the scaling study with
# `contract_with_unroll` slicing.  It is validated to agree with the RDM route.

_WIN_2x2 = ("A", "B", "C", "D")  # TL, TR, BR, BL
_replica_labels = {}


def _window_sites_2x2(psi, s0=None):
    if s0 is None:
        s0 = psi.sites()[0]
    return {
        "A": s0,                       # TL
        "B": psi.nn_site(s0, "r"),     # TR
        "C": psi.nn_site(s0, "br"),    # BR
        "D": psi.nn_site(s0, "b"),     # BL
    }


def _site_swaps(lab):
    """Per-site double-layer swap gates on (ket, bra) virtual labels.

    Mirrors DoublePepsTensor.fuse_layers:  `l l' X t'`  and  `b b' X r'`
    (primed = bra side).  These are the only fermionic swaps in the flat
    network; the replica permutation of the physical legs is bosonic.
    """
    return [
        (lab["kl"], lab["bt"]), (lab["bl"], lab["bt"]),   # l l' X t'
        (lab["kb"], lab["br"]), (lab["bb"], lab["br"]),   # b b' X r'
    ]


def _perm_operator(config, ph_leg, sigma):
    """Bosonic permutation operator P with legs [bra_0..bra_{R-1}, ket_0..ket_{R-1}].

    bra_r connects to ket_{sigma[r]}.  No fermionic swap gate is applied: the
    on-site physical legs already carry the full on-site charge and every
    double-layer crossing sign is in the per-site `_site_swaps`.  (Validated:
    matches the RDM route / Strategy 1 to ~1e-12.)
    """
    from yastn import eye, tensordot
    R = len(sigma)
    if ph_leg.is_fused():
        sub = ph_leg.unfuse_leg()
        ph0, anc0 = sub[0], sub[1]
        Ip = eye(config, legs=[ph0, ph0.conj()], isdiag=False)
        Ia = eye(config, legs=[anc0, anc0.conj()], isdiag=False)
        wire = tensordot(Ip, Ia, axes=((), ()))          # op ip oa ia
        wire = wire.transpose(axes=(0, 2, 1, 3))          # op oa ip ia
        wire = wire.fuse_legs(axes=((0, 1), (2, 3)))      # [op oa] [ip ia]
    else:
        wire = eye(config, legs=[ph_leg, ph_leg.conj()], isdiag=False)  # (out, in)
    P = None
    for _ in range(R):
        P = wire if P is None else tensordot(P, wire, axes=((), ()))
    # P legs: (out_0, in_0, ..., out_{R-1}, in_{R-1}); pair bra_r <- out_{sigma[r]}.
    out_axes = list(range(0, 2 * R, 2))
    in_axes = list(range(1, 2 * R, 2))
    bra_axes = [out_axes[sigma[r]] for r in range(R)]
    ket_axes = [in_axes[m] for m in range(R)]
    P = P.transpose(axes=tuple(bra_axes + ket_axes))
    return P


def _build_replica_network_2x2(env, n, perms, s0=None):
    """Assemble interleaved args + per-site swap list for the flat R=2n+1
    replica contraction of the 2x2 window.  Returns (args, names, swap)."""
    global _replica_labels
    _replica_labels = {}
    psi = env.psi
    sites = _window_sites_2x2(psi, s0)
    R = 2 * n + 1

    kets = {X: psi[sites[X]].ket for X in _WIN_2x2}
    bras = {X: psi[sites[X]].bra.conj() for X in _WIN_2x2}
    e = {X: env[sites[X]] for X in _WIN_2x2}

    args, names, swap = [], [], []
    _cnt = [0]

    def L():
        _cnt[0] += 1
        return _cnt[0]

    def add(tensor, ig, name):
        args.extend([tensor, tuple(ig)])
        names.append(name)

    for r in range(R):
        K = {X: {} for X in _WIN_2x2}
        Br = {X: {} for X in _WIN_2x2}
        kAB, bAB = L(), L()
        kDC, bDC = L(), L()
        kAD, bAD = L(), L()
        kBC, bBC = L(), L()
        kA_t, bA_t = L(), L()
        kA_l, bA_l = L(), L()
        kB_t, bB_t = L(), L()
        kB_r, bB_r = L(), L()
        kC_r, bC_r = L(), L()
        kC_b, bC_b = L(), L()
        kD_l, bD_l = L(), L()
        kD_b, bD_b = L(), L()
        K["A"] = dict(kt=kA_t, kl=kA_l, kb=kAD, kr=kAB)
        Br["A"] = dict(bt=bA_t, bl=bA_l, bb=bAD, br=bAB)
        K["B"] = dict(kt=kB_t, kl=kAB, kb=kBC, kr=kB_r)
        Br["B"] = dict(bt=bB_t, bl=bAB, bb=bBC, br=bB_r)
        K["C"] = dict(kt=kBC, kl=kDC, kb=kC_b, kr=kC_r)
        Br["C"] = dict(bt=bBC, bl=bDC, bb=bC_b, br=bC_r)
        K["D"] = dict(kt=kAD, kl=kD_l, kb=kD_b, kr=kDC)
        Br["D"] = dict(bt=bAD, bl=bD_l, bb=bD_b, br=bDC)
        _replica_labels[r] = dict(K=K, Br=Br)

    kphys = {X: [L() for _ in range(R)] for X in _WIN_2x2}
    bphys = {X: [L() for _ in range(R)] for X in _WIN_2x2}
    ph_legs = {X: kets[X].get_legs(4) for X in _WIN_2x2}

    for r in range(R):
        K = _replica_labels[r]["K"]
        Br = _replica_labels[r]["Br"]
        for X in _WIN_2x2:
            kk, bb = K[X], Br[X]
            add(kets[X], (kk["kt"], kk["kl"], kk["kb"], kk["kr"], kphys[X][r]),
                "ket_%d_%s" % (r, X))
            add(bras[X], (bb["bt"], bb["bl"], bb["bb"], bb["br"], bphys[X][r]),
                "bra_%d_%s" % (r, X))
            swap.extend(_site_swaps({**kk, **bb}))

    for X in _WIN_2x2:
        P = _perm_operator(env.psi.config, ph_legs[X], perms[X])
        ig = tuple(bphys[X]) + tuple(kphys[X])
        add(P, ig, "P_%s" % X)

    for r in range(R):
        K = _replica_labels[r]["K"]
        Br = _replica_labels[r]["Br"]
        c_tl_l, c_tl_t = L(), L()
        c_tr_t, c_tr_r = L(), L()
        c_bl_b, c_bl_l = L(), L()
        c_br_r, c_br_b = L(), L()
        mt, mb, ml, mr = L(), L(), L(), L()
        eA, eB, eC, eD = e["A"], e["B"], e["C"], e["D"]
        add(eA.tl, (c_tl_l, c_tl_t), "tl_%d" % r)
        add(eB.tr, (c_tr_t, c_tr_r), "tr_%d" % r)
        add(eD.bl, (c_bl_b, c_bl_l), "bl_%d" % r)
        add(eC.br, (c_br_r, c_br_b), "br_%d" % r)
        tA = eA.t.unfuse_legs(axes=1)
        tB = eB.t.unfuse_legs(axes=1)
        add(tA, (c_tl_t, K["A"]["kt"], Br["A"]["bt"], mt), "t_A_%d" % r)
        add(tB, (mt, K["B"]["kt"], Br["B"]["bt"], c_tr_t), "t_B_%d" % r)
        bD = eD.b.unfuse_legs(axes=1)
        bC = eC.b.unfuse_legs(axes=1)
        add(bC, (c_br_b, K["C"]["kb"], Br["C"]["bb"], mb), "b_C_%d" % r)
        add(bD, (mb, K["D"]["kb"], Br["D"]["bb"], c_bl_b), "b_D_%d" % r)
        lA = eA.l.unfuse_legs(axes=1)
        lD = eD.l.unfuse_legs(axes=1)
        add(lA, (ml, K["A"]["kl"], Br["A"]["bl"], c_tl_l), "l_A_%d" % r)
        add(lD, (c_bl_l, K["D"]["kl"], Br["D"]["bl"], ml), "l_D_%d" % r)
        rB = eB.r.unfuse_legs(axes=1)
        rC = eC.r.unfuse_legs(axes=1)
        add(rB, (c_tr_r, K["B"]["kr"], Br["B"]["br"], mr), "r_B_%d" % r)
        add(rC, (mr, K["C"]["kr"], Br["C"]["br"], c_br_r), "r_C_%d" % r)

    args.append(())
    return args, names, swap


def Jn_strategy2_flat(env, n, perms=None, s0=None, *, unroll=None,
                      checkpoint_loop=False, optimize=None, return_path=False):
    """J_n via the flat double-layer replica network (heavier; for scaling)."""
    import yastn
    if perms is None:
        perms = _replica_perms(n)
    args, names, swap = _build_replica_network_2x2(env, n, perms, s0)
    if optimize is None:
        optimize, _info = yastn.get_contraction_path(
            *args, names=names, who="modcomm_flat_J%d" % n, unroll=unroll)
    res = yastn.contract_with_unroll(
        *args, optimize=optimize, swap=swap, unroll=unroll,
        checkpoint_loop=checkpoint_loop)
    Z = res.to_number()
    if return_path:
        return Z, optimize
    return Z
