"""JSON I/O for yastn `to_dict` / `from_dict` payloads.

Encoder + decoder are the same pair used in
``tn-torch_dev_square/ipeps/tensor_io.py`` (``NumPy_Encoder``) and
``tn-torch_dev_square/ipeps/integration_yastn.py`` (``complex_decoder``),
so JSONs produced here stay compatible with that codebase's reader. We
additionally apply a post-load int-key fixup because yastn's ``site_data``
uses int keys (the output of ``Geometry.site2index``) which JSON forces to
strings on the round-trip.
"""
import json
import numpy as np


class NumPy_Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, complex):  # type complex is not json-serializable
            return {"real": obj.real, "imag": obj.imag}
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumPy_Encoder, self).default(obj)


def complex_decoder(dct):
    # Inverse of the {"real": ..., "imag": ...} form emitted for complex scalars.
    if "real" in dct and "imag" in dct:
        return complex(dct["real"], dct["imag"])
    return dct


def _fix_int_keys(obj):
    """Convert digit-string dict keys back to int after a JSON round-trip.

    JSON forces dict keys to strings, but yastn's `site_data` uses int keys
    (from ``Geometry.site2index``); without this, ``net._site_data[k] = ...``
    in ``Peps.from_dict`` would create string keys that subsequent
    ``__getitem__`` lookups (which use int) silently miss. Safe for our
    schema because no inner dict legitimately uses digit-only string keys.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            nk = int(k) if isinstance(k, str) and k.lstrip('-').isdigit() else k
            out[nk] = _fix_int_keys(v)
        return out
    if isinstance(obj, list):
        return [_fix_int_keys(x) for x in obj]
    return obj


def dump(obj, f):
    json.dump(obj, f, cls=NumPy_Encoder)


def load(f):
    return _fix_int_keys(json.load(f, object_hook=complex_decoder))
