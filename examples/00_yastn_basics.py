"""
00 — yastn tensor basics: initialization and contraction.

A runnable walkthrough of the low-level yastn API that underlies the rest of
this example. Nothing here touches PEPS / CTM — it only builds a couple of
block-sparse symmetric tensors and contracts them a few different ways.

Prints a brief block summary after each step so you can see which charge
sectors survive the symmetry selection rule.
"""
import yastn


def make_tensors():
    # Z2 symmetry, numpy backend (default), complex128. Matches the config
    # used by scripts 01-03 so the patterns you see here carry over.
    config = yastn.make_config(sym="Z2", fermionic=True, default_dtype="complex128")

    # A Leg specifies: signature s (+1 ingoing / -1 outgoing), the list of
    # charge sectors t present on the leg, and the block dimension D for each.
    leg_in = yastn.Leg(config, s=+1, t=(0, 1), D=(2, 2))
    leg_out = yastn.Leg(config, s=-1, t=(0, 1), D=(2, 2))

    # Rank-3 tensor with signature (+1, -1, -1), filled with random values in
    # every block allowed by charge conservation (sum_i s_i * t_i = 0 mod 2).
    a = yastn.rand(config=config, legs=[leg_in, leg_out, leg_out])

    # Alternative construction: empty tensor + hand-set blocks. yastn rejects
    # any ts tuple that violates charge conservation, which is how symmetry
    # bugs get caught at build time.
    b = yastn.Tensor(config=config, s=(+1, -1, -1))
    b.set_block(ts=(0, 0, 0), Ds=(2, 2, 2), val="rand")
    b.set_block(ts=(1, 1, 0), Ds=(2, 2, 2), val="rand")
    b.set_block(ts=(1, 0, 1), Ds=(2, 2, 2), val="rand")
    b.set_block(ts=(0, 1, 1), Ds=(2, 2, 2), val="rand")
    return config, leg_in, leg_out, a, b


def describe(name, t):
    """One-line summary: rank, signature, and the allowed (ts, Ds) blocks."""
    print(f"  {name}: rank={t.ndim}, s={t.s}, n_blocks={len(t.get_blocks_charge())}")
    for ts, Ds in zip(t.get_blocks_charge(), t.get_blocks_shape()):
        print(f"    ts={ts}  Ds={Ds}")


def contraction_examples(config, leg_out, a, b):
    # Contraction requires matching dims AND opposite signatures on the
    # contracted legs. Build `c` with a leading ingoing leg (leg_out.conj())
    # so we can contract it into a's last leg.
    c = yastn.rand(config=config, legs=[leg_out.conj(), leg_out, leg_out])

    # tensordot: contract axis 2 of a with axis 0 of c. Remaining legs keep
    # their relative order: (a's 0, a's 1, c's 1, c's 2).
    ac = yastn.tensordot(a, c, axes=(2, 0))

    # `@` is the shorthand "last leg of LHS × first leg of RHS" — same result.
    ac_at = a @ c
    assert yastn.norm(ac - ac_at) < 1e-12

    # ncon / einsum give explicit index-label control. Negative labels mark
    # the open legs of the result (in the order of decreasing magnitude, as
    # per ncon convention); positive labels mark contracted pairs.
    ac_ncon = yastn.ncon([a, c], ((-0, -1, 1), (1, -2, -3)))
    ac_eins = yastn.einsum("ijx,xkl->ijkl", a, c)
    assert yastn.norm(ac - ac_ncon) < 1e-12
    assert yastn.norm(ac - ac_eins) < 1e-12

    # Full contraction: vdot conjugates its first argument and contracts over
    # every leg, returning a backend scalar directly.
    norm_sq = yastn.vdot(a, a)
    return ac, norm_sq


if __name__ == "__main__":
    config, leg_in, leg_out, a, b = make_tensors()
    print("Initialised Z2-symmetric tensors:")
    describe("a (rand)", a)
    describe("b (set_block)", b)

    ac, norm_sq = contraction_examples(config, leg_out, a, b)
    print("\nContraction  tensordot(a, c, axes=(2, 0))  ->")
    describe("a @ c", ac)
    print(f"\n<a|a> = {norm_sq:.6f}")
