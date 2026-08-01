"""Guards for the unified force-field registry (`_FORCE_FIELDS`).

Checks that the registry is well-formed, that names don't collide, that name
resolution accepts ids/labels/aliases, and that the RS-facing catalog shape is
intact. These are pure-metadata checks: they need neither psi4 nor tblite
installed (the `available` flags simply reflect whatever is present).
"""
import pytest

from chemdm.potentialInterface import (
    DEFAULT_FORCE_FIELD,
    _FORCE_FIELDS,
    available_force_fields,
    resolve_force_field,
)


def test_registry_entries_are_well_formed():
    for ff in _FORCE_FIELDS:
        assert {"id", "label", "category", "backend", "supports_tp", "build"} <= set(ff)
        assert ff["backend"] in {"tblite", "psi4"}
        if ff["backend"] == "tblite":
            assert "method" in ff["build"]
        else:
            assert "functional" in ff["build"]
            assert ff["build"]["reference"] in {"ks", "hf"}


def test_names_resolve_unambiguously():
    # A name may repeat within one entry (an xTB label lower-cases to its id),
    # but must never point at two different force fields.
    mapping = {}
    for ff in _FORCE_FIELDS:
        names = {ff["id"].lower(), ff["label"].lower(),
                 *(a.lower() for a in ff.get("aliases", []))}
        for n in names:
            assert mapping.setdefault(n, ff["id"]) == ff["id"], \
                f"name {n!r} is claimed by two force fields"


def test_resolve_accepts_id_label_and_alias():
    assert resolve_force_field("gfn2-xtb") == "gfn2-xtb"            # canonical id
    assert resolve_force_field("GFN2-xTB") == "gfn2-xtb"            # display label
    assert resolve_force_field("ωB97X-D / def2-TZVP") == "wb97x-d"  # display label
    assert resolve_force_field("wb97x-d") == "wb97x-d"             # id
    assert resolve_force_field("xtb2") == "gfn2-xtb"               # alias normalizes to id


def test_resolve_unknown_raises():
    with pytest.raises(ValueError):
        resolve_force_field("does-not-exist")


def test_default_force_field_is_a_registry_id():
    assert resolve_force_field(DEFAULT_FORCE_FIELD) == DEFAULT_FORCE_FIELD
    assert DEFAULT_FORCE_FIELD in {ff["id"] for ff in _FORCE_FIELDS}


def test_available_force_fields_shape():
    ffs = available_force_fields()
    assert len(ffs) == len(_FORCE_FIELDS)
    required = {"id", "label", "category", "supports_tp", "available", "reason"}
    for ff in ffs:
        assert required <= set(ff), f"missing keys in {ff}"
        assert isinstance(ff["supports_tp"], bool)
        assert isinstance(ff["available"], bool)
        # An unavailable backend must explain why; an available one need not.
        assert ff["available"] or ff["reason"]
