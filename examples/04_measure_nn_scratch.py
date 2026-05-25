"""
04 — Horizontal NN bond <c^+_s0 c_s1>, built from scratch with the
oe_blocksparse one-shot einsum on the yastn ``oe_blocksparse`` branch.

Ket and bra are kept as separate tensors; the env edges' bulk-facing legs are
un-fused so the double-layer structure is not pre-combined. The full 14-tensor
network (4 corners + 4 edges + ket0 + bra0 + ket1 + bra1) is handed to
``yastn.contract_with_unroll`` in one call — opt_einsum picks the pairwise
contraction path.

Leg storage (from yastn.tn.fpeps.envs._env_contractions):

    corners  (2 legs, both fused [bra,ket])         edges  (3 legs, each fused)
    tl: (down,  right)                              t: (left, down,  right)
    tr: (left, down)                                b: (right, up,   left)
    bl: (right, up)                                 l: (down, right, up)
    br: (up,   left)                                r: (up,  left,   down)

    Each fused leg unfuses as (bra, ket) in that order.

    on-site (ket/bra after apply_gate_on_ket + separated)
        ket: (t, l, b, r, s)        — operator absorbed;  one virtual leg
                                      (r for cp at s0, l for c at s1) is
                                      fused with the aux leg of the op.
        bra: (t, l, b, r, s)        — unchanged peps; gets .conj() before
                                      being handed to the einsum.

============================================================================
CONTRACTION DIAGRAM — external CTM + swap-gate crossings inside A0
============================================================================

(a) External network (boundaries are single fused bonds, bulk legs are split
    into a ket line and a bra line):

      [tl]────1────[t0]─────2─────[t1]────3────[tr]
        │           │ │          │ │          │
        4           ║ ║          ║ ║          6
        │          14 13        20 19         │
        │           ║ ║          ║ ║          │
       [l]─────12══════11══17═══════ket1══23─[r]
        │           ║ ║      ═18═ ║ ║         │
        │          16 15        22 21        24
        │           ║ ║          ║ ║          │
        5           ║ ║          ║ ║          7
        │           ║ ║          ║ ║          │
      [bl]────8───[b0]─────9─────[b1]───10───[br]

    Fused CTM-boundary bonds (1..10)            single line
    Bulk bonds split ket/bra (11..24)           double line ═
    A0-A1 ket bond 17 carries the fused aux leg
    Physical ket↔bra bonds 25 (A0), 26 (A1)     not shown above

(b) Internals of A0 with explicit swap-gate crossings. Both swap gates live
    inside the ket0 tensor after operator absorption:

                 t_ket (= bond 13)
                   │
            ┌──────┴──────┐
            │    ket(A0)  │
            │             │
       l────┤   phys      ├────╲              ╱──── aux (exits to bond 17,
      (11)  │    │        │     ╲            ╱       then contracts directly
            │  [cp]       │      ╲          ╱        with aux of c in ket1)
            │   ╱ ╲       │       ╲        ╱
            │  p   a      │        ╲      ╱
            │   ╲ ╱       │         ╲    ╱
            │    ╳   ←SG1 │          ╲  ╱
            │   ╱ ╲       │           ╲╱
            │  a   p      │           ╳       ← SG2
            │  │   │      │           ╱╲
            │  │   └──────┤ (→ ket.s) ╱  ╲
            │  │          │          ╱    ╲
            │ aux ────────┼─────────╱      ╲
            │             │                ╲
            │             │
            │   b_ket ────┤────── (crosses aux above at SG2)
            │             │
            └──────┬──────┘
                   │
                 b_ket (= bond 15)

    Legend: ╳ with two crossing lines = swap_gate on the two crossed legs.

(c) Additional swap gates from the double-layer convention (normally hidden
    inside DoublePepsTensor.fuse_layers). With ket/bra kept separate, these
    are fed explicitly to ncon via the ``swap`` kwarg:

        at site 0:  swap(ket0.l , bra0.t) ,  swap(bra0.l , bra0.t)
                    swap(ket0.b , bra0.r) ,  swap(bra0.b , bra0.r)
        at site 1:  swap(ket1.l , bra1.t) ,  swap(bra1.l , bra1.t)
                    swap(ket1.b , bra1.r) ,  swap(bra1.b , bra1.r)
"""
import os

import yastn
from yastn.tn.fpeps import Bond
from yastn.tn.fpeps._gates_auxiliary import gate_fix_swap_gate

import _jsonio

HERE = os.path.dirname(__file__)
ENV_PATH = os.path.join(HERE, "out", "ci_env_chi.json")


def load_env_and_ops(sym="Z2"):
    config = yastn.make_config(sym=sym, fermionic=True, default_dtype="complex128")
    with open(ENV_PATH, "r") as f:
        env = yastn.from_dict(_jsonio.load(f), config=config)
    ops = yastn.operators.SpinlessFermions(sym=sym, backend=config.backend,
                                           default_dtype="complex128")
    return env, ops


# --------------------------------------------------------------------------- #
#  Step 0 — operator prep + absorb (SG1 inside cp, SG2 inside A0.ket)         #
# --------------------------------------------------------------------------- #

def prepare_ops(cp, c, env, bond):
    cp3 = cp.add_leg(s=+1, axis=2)
    c3  =  c.add_leg(s=-1, axis=2)
    cp3 = cp3.swap_gate(axes=(1, 2))       # SG1
    dirn = env.nn_bond_dirn(*bond)
    cp3, c3 = gate_fix_swap_gate(cp3, c3, dirn, env.f_ordered(*bond))
    return cp3, c3


def absorb_ops(env, bond, cp3, c3):
    s0, s1 = bond
    A0 = env.psi[s0].apply_gate_on_ket(cp3, dirn='l')   # SG2 baked inside ket
    A1 = env.psi[s1].apply_gate_on_ket(c3,  dirn='r')
    return A0, A1


# --------------------------------------------------------------------------- #
#  One-shot einsum — ket/bra separated, env bulk legs un-fused                #
# --------------------------------------------------------------------------- #

def contract_one_shot(A0, A1, e0, e1, who="measure_cpc_nn_h"):
    # Un-fuse each env edge's bulk-facing leg (axis 1) into (bra, ket).
    t0 = e0.t.unfuse_legs(axes=1)
    t1 = e1.t.unfuse_legs(axes=1)
    b0 = e0.b.unfuse_legs(axes=1)
    b1 = e1.b.unfuse_legs(axes=1)
    l  = e0.l.unfuse_legs(axes=1)
    r  = e1.r.unfuse_legs(axes=1)
    tl, bl = e0.tl, e0.bl
    tr, br = e1.tr, e1.br

    # ket / bra tensors — bra gets .conj() (matching fuse_layers: Ak · Ab.conj()).
    ket0, bra0 = A0.ket, A0.bra.conj()
    ket1, bra1 = A1.ket, A1.bra.conj()

    # Env edges store their middle leg as [bra_side, ket_conj_side] (unfuse
    # order). The "bra_side" (un-conjugated in env construction) must pair
    # with our un-conjugated bulk layer = ket; the "ket_conj_side" must pair
    # with our conjugated bulk layer = bra.conj().
    args = (
        # --- env corners (fused bonds only, nothing to unfuse) ---
        tl,   (4, 1),
        t0,   (1, 14, 13, 2),
        t1,   (2, 20, 19, 3),
        tr,   (3, 6),
        l,    (5, 12, 11, 4),
        r,    (6, 24, 23, 7),
        bl,   (8, 5),
        b0,   (9, 16, 15, 8),
        b1,   (10, 22, 21, 9),
        br,   (7, 10),
        # --- bulk ket / bra ---           t    l    b    r    s
        ket0,                           (14,  12,  16,  18,  25),
        bra0,                           (13,  11,  15,  17,  25),
        ket1,                           (20,  18,  22,  24,  26),
        bra1,                           (19,  17,  21,  23,  26),
        (),
    )
    names = ('tl', 't0', 't1', 'tr', 'l', 'r', 'bl', 'b0', 'b1', 'br',
             'ket0', 'bra0', 'ket1', 'bra1')

    # fuse_layers' internal swap gates, translated to the unfused labels.
    swap = [
        (12, 13), (11, 13), (16, 17), (15, 17),   # site 0
        (18, 19), (17, 19), (22, 23), (21, 23),   # site 1
    ]

    path, _ = yastn.get_contraction_path(*args, names=names, who=who)
    return yastn.contract_with_unroll(*args, optimize=path, swap=swap).to_number()


def measure_cp_c_horizontal(env, ops, site=(0, 0)):
    s0 = site
    s1 = env.psi.nn_site(s0, 'r')
    bond = Bond(s0, s1)
    if env.nn_bond_dirn(*bond) not in ('lr', 'rl'):
        raise ValueError(f"{s0}->{s1} is not a horizontal NN bond.")

    cp3, c3 = prepare_ops(ops.cp(), ops.c(), env, bond)
    A0_op, A1_op = absorb_ops(env, bond, cp3, c3)

    # denominator reuses psi[s0], psi[s1] as their own "ket" (no op applied).
    A0_no = env.psi[s0]
    A1_no = env.psi[s1]

    e0, e1 = env[s0], env[s1]
    num = contract_one_shot(A0_op, A1_op, e0, e1, who="num")
    den = contract_one_shot(A0_no, A1_no, e0, e1, who="den")
    return num / den


if __name__ == "__main__":
    env, ops = load_env_and_ops()
    val = measure_cp_c_horizontal(env, ops, site=(0, 0))
    ref = env.measure_nn(ops.cp(), ops.c(),
                         bond=Bond((0, 0), env.psi.nn_site((0, 0), 'r')))
    assert abs(val - ref) < 1e-12, (val, ref, abs(val - ref))
