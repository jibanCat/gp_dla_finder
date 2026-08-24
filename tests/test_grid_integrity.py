"""The sidecar must describe the grid it sits beside, not merely resemble one.

A sidecar carrying fields with the right names proves nothing on its own. It
could have been copied from a different grid, or the arrays edited after it was
written, and every field name would still look correct. So the recorded digests
are recomputed at load time and compared against the file and the arrays.

The consequence is deliberate: a grid whose sidecar does not check out loads
fine for inspection and is refused for inference.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gp_dla_finder import Config, available_sample_grids, load_sample_grid
from gp_dla_finder.finder import Finder, SampleGridMismatch
from gp_dla_finder.samples import (
    canonical_array_digest,
    canonical_file_digest,
    verify_grid_integrity,
)

ROOT = Path(__file__).resolve().parents[1]
PACKAGED = ROOT / "src" / "gp_dla_finder" / "data" / "samples"


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """A grid from the supported builder, arrays and sidecar together."""
    out = tmp_path_factory.mktemp("n89")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "build_sample_grid.py"),
            "--name",
            "integrity_probe_1200",
            "--num-samples",
            "1200",
            "--log-nhi",
            "20.3",
            "22.5",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    return out / "integrity_probe_1200.npz"


def _copy(built: Path, into: Path) -> Path:
    """The grid and its sidecar, somewhere writable."""
    into.mkdir(parents=True, exist_ok=True)
    npz = into / built.name
    shutil.copy(built, npz)
    shutil.copy(built.with_suffix(".json"), npz.with_suffix(".json"))
    return npz


def _rewrite_sidecar(npz: Path, mutate) -> None:
    sidecar = npz.with_suffix(".json")
    payload = json.loads(sidecar.read_text())
    mutate(payload)
    sidecar.write_text(json.dumps(payload, indent=2) + "\n")


def _refused(npz: Path) -> tuple[str, ...]:
    grid = load_sample_grid(path=npz)
    assert grid.usable_for_inference is False
    with pytest.raises(SampleGridMismatch):
        Finder(
            Config.desi_y3(max_absorbers=1, sample_grid=grid.name),
            grid=grid,
            warn_about_threads=False,
        )
    return grid.unusable_because


# --- the supported path still works ------------------------------------------


def test_the_builders_own_grid_passes_intact(built):
    grid = load_sample_grid(path=built)
    assert grid.integrity_problems == ()
    assert grid.usable_for_inference is True
    assert grid.name == "integrity_probe_1200"


@pytest.mark.parametrize("name", sorted(available_sample_grids()))
def test_every_packaged_grid_passes_its_own_integrity_check(name):
    """The bundled assets go through the same path as an external grid.

    A shipped grid that had been rebuilt without refreshing its sidecar, or
    corrupted in transit, would otherwise be the one case nothing checked.
    """
    grid = load_sample_grid(name)
    assert grid.integrity_problems == (), grid.integrity_problems
    assert grid.usable_for_inference is True


def test_the_digest_definition_is_shared_with_the_builder():
    """One definition, imported -- not two that agree today.

    Two independent notions of "the hash of this array" would eventually
    disagree over dtype, byte order or contiguity, and then every correctly
    built grid would fail its own check.
    """
    source = (ROOT / "tools" / "build_sample_grid.py").read_text()
    assert "canonical_array_digest as sha256_array" in source
    assert "canonical_file_digest as sha256_file" in source
    assert "hashlib.sha256" not in source, "the builder redefines a digest"


# --- and every way it can fail ------------------------------------------------


def test_a_changed_array_value_is_refused(built, tmp_path):
    """One number edited inside the .npz.

    ``offset_samples`` is mutated because it carries no cross-array invariant.
    Editing ``log_nhi_samples`` is caught even earlier -- see the next test --
    which is a different mechanism and worth testing separately rather than
    conflating with the digest.
    """
    npz = _copy(built, tmp_path / "edited")
    with np.load(npz) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["offset_samples"][7] = float(arrays["offset_samples"][7]) + 1e-9
    np.savez_compressed(npz, **arrays)

    reasons = _refused(npz)
    assert any("arrays.offset_samples" in r for r in reasons)
    assert any("sha256" in r for r in reasons)


def test_an_edit_that_breaks_a_semantic_invariant_is_caught_even_earlier(
    built, tmp_path
):
    """Two independent defences, and the older one fires first.

    ``nhi_samples`` and ``10**log_nhi_samples`` must agree, so editing one of
    them alone makes the grid refuse to CONSTRUCT -- before any digest is
    compared. The digest check is what catches an edit that keeps the arrays
    mutually consistent, which this one does not.
    """
    npz = _copy(built, tmp_path / "inconsistent")
    with np.load(npz) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["log_nhi_samples"][7] += 1e-6
    np.savez_compressed(npz, **arrays)

    with pytest.raises(ValueError, match="not consistent with"):
        load_sample_grid(path=npz)


def test_a_consistent_edit_to_both_column_density_arrays_is_still_refused(
    built, tmp_path
):
    """The case the semantic invariant cannot see, and the digest can.

    Editing log_nhi_samples AND nhi_samples together keeps them consistent, so
    the grid constructs -- and the recorded digests are what notice.
    """
    npz = _copy(built, tmp_path / "consistent-edit")
    with np.load(npz) as data:
        arrays = {k: data[k].copy() for k in data.files}
    arrays["log_nhi_samples"][7] += 1e-6
    arrays["nhi_samples"][7] = 10.0 ** arrays["log_nhi_samples"][7]
    np.savez_compressed(npz, **arrays)

    reasons = _refused(npz)
    assert any("arrays.log_nhi_samples" in r for r in reasons)
    assert any("arrays.nhi_samples" in r for r in reasons)


def test_a_sidecar_from_a_different_grid_is_refused(tmp_path):
    """The exact failure a name-only check cannot see."""
    npz = tmp_path / "swapped.npz"
    shutil.copy(PACKAGED / "pw14_172_225_10000.npz", npz)
    shutil.copy(PACKAGED / "pw14_172_225_50000.json", npz.with_suffix(".json"))

    reasons = _refused(npz)
    assert any("sha256" in r for r in reasons)
    assert any("shape" in r for r in reasons)


def test_a_false_file_digest_is_refused(built, tmp_path):
    npz = _copy(built, tmp_path / "badfile")
    _rewrite_sidecar(npz, lambda p: p.__setitem__("sha256", "0" * 64))

    reasons = _refused(npz)
    assert any(r.startswith("integrity: sha256:") for r in reasons)


def test_a_malformed_file_digest_is_refused(built, tmp_path):
    npz = _copy(built, tmp_path / "nofile")
    _rewrite_sidecar(npz, lambda p: p.__setitem__("sha256", ""))

    reasons = _refused(npz)
    assert any("no usable file digest" in r for r in reasons)


def test_a_false_per_array_digest_is_refused(built, tmp_path):
    """The file hash still matches; only one array's digest is wrong."""
    npz = _copy(built, tmp_path / "badarray")

    def mutate(payload):
        payload["arrays"]["nhi_samples"]["sha256_float64"] = "1" * 64

    _rewrite_sidecar(npz, mutate)

    reasons = _refused(npz)
    assert any("arrays.nhi_samples" in r for r in reasons)
    # And the file digest is NOT the thing that caught it.
    assert not any(r.startswith("integrity: sha256:") for r in reasons)


def test_a_declared_sample_count_that_disagrees_is_refused(built, tmp_path):
    npz = _copy(built, tmp_path / "badcount")

    def mutate(payload):
        payload["qmc"]["num_samples"] = 999

    _rewrite_sidecar(npz, mutate)

    reasons = _refused(npz)
    assert any("qmc.num_samples" in r for r in reasons)


def test_a_missing_array_entry_is_refused(built, tmp_path):
    npz = _copy(built, tmp_path / "noarray")

    def mutate(payload):
        del payload["arrays"]["offset_samples"]

    _rewrite_sidecar(npz, mutate)

    reasons = _refused(npz)
    assert any("has no record for it" in r for r in reasons)


def test_a_malformed_sidecar_is_refused_rather_than_ignored(built, tmp_path):
    """Not the same as having no sidecar.

    A file that claims to describe this grid and cannot be parsed is a fault,
    not an absence, and silently falling back to "no provenance" would hide it.
    """
    npz = _copy(built, tmp_path / "brokenjson")
    npz.with_suffix(".json").write_text("{ this is not json")

    grid = load_sample_grid(path=npz)
    assert grid.usable_for_inference is False
    assert any("not valid JSON" in r for r in grid.unusable_because)


def test_a_bare_npz_is_never_verified(built, tmp_path):
    """No sidecar means the check never ran, which is not the same as passing."""
    npz = tmp_path / "bare" / built.name
    npz.parent.mkdir()
    shutil.copy(built, npz)

    grid = load_sample_grid(path=npz)
    assert grid.integrity_problems is None
    assert grid.usable_for_inference is False
    assert any("never checked" in r for r in grid.unusable_because)


# --- inspection stays available ----------------------------------------------


def test_a_failing_grid_still_loads_for_inspection(built, tmp_path):
    """Refusal is at inference, not at load."""
    npz = _copy(built, tmp_path / "inspect")
    _rewrite_sidecar(npz, lambda p: p.__setitem__("sha256", "0" * 64))

    grid = load_sample_grid(path=npz)
    assert grid.num_samples == 1200
    assert np.all(grid.log_nhi_samples >= 20.3)
    assert grid.usable_for_inference is False


def test_inspection_cannot_upgrade_a_grid(built, tmp_path):
    """Reading the arrays must not mark anything as verified."""
    npz = tmp_path / "bare2" / built.name
    npz.parent.mkdir()
    shutil.copy(built, npz)

    grid = load_sample_grid(path=npz)
    _ = grid.log_nhi_samples, grid.nhi_samples, grid.num_samples, grid.log_nhi_range
    assert grid.integrity_problems is None
    assert grid.usable_for_inference is False


# --- the checker itself -------------------------------------------------------


def test_verify_reports_every_problem_not_only_the_first(built, tmp_path):
    """A caller fixing one fault should see the others in the same message."""
    npz = _copy(built, tmp_path / "several")

    def mutate(payload):
        payload["sha256"] = "0" * 64
        payload["qmc"]["num_samples"] = 5
        payload["arrays"]["nhi_samples"]["sha256_float64"] = "2" * 64

    _rewrite_sidecar(npz, mutate)

    with np.load(npz) as data:
        arrays = {k: data[k] for k in data.files}
    problems = verify_grid_integrity(
        npz, json.loads(npz.with_suffix(".json").read_text()), arrays
    )
    assert len(problems) >= 3


def test_the_canonical_digests_are_stable():
    """Pinned, because the recorded hashes in every sidecar depend on them."""
    array = np.array([1.0, 2.0, 3.0])
    assert canonical_array_digest(array) == canonical_array_digest(
        np.asarray(array, dtype=np.float32).astype(np.float64)
    )
    # A non-contiguous view hashes as its values, not its memory layout.
    assert canonical_array_digest(np.arange(6.0)[::2]) == canonical_array_digest(
        np.array([0.0, 2.0, 4.0])
    )


def test_the_file_digest_matches_a_known_value(tmp_path):
    probe = tmp_path / "probe.bin"
    probe.write_bytes(b"gp_dla_finder")
    import hashlib

    assert canonical_file_digest(probe) == hashlib.sha256(b"gp_dla_finder").hexdigest()


# --- a sidecar can be valid JSON and still not describe a grid ----------------
#
# Every one of these used to escape as a raw AttributeError, TypeError, KeyError
# or ValueError from inside a load. A crash that happens to prevent inference is
# not an integrity check: it does not say which field is wrong, and it cannot be
# reported alongside the other faults a caller has to fix.


def test_a_missing_shape_is_refused(built, tmp_path):
    """Required, not optional.

    A sidecar that omits the shape has not described the array, and accepting
    the omission would let the weakest sidecar through the same gate as a
    complete one.
    """
    npz = _copy(built, tmp_path / "noshape")

    def mutate(payload):
        del payload["arrays"]["log_nhi_samples"]["shape"]

    _rewrite_sidecar(npz, mutate)

    reasons = _refused(npz)
    assert any("arrays.log_nhi_samples.shape: missing" in r for r in reasons)


def test_a_wrong_shape_is_refused(built, tmp_path):
    npz = _copy(built, tmp_path / "wrongshape")

    def mutate(payload):
        payload["arrays"]["offset_samples"]["shape"] = [999]

    _rewrite_sidecar(npz, mutate)

    reasons = _refused(npz)
    assert any("arrays.offset_samples.shape" in r and "999" in r for r in reasons)


@pytest.mark.parametrize(
    "shape",
    [42, "1200", {"n": 1200}, [1200, "x"], None],
    ids=["int", "string", "object", "mixed-list", "null"],
)
def test_a_malformed_shape_is_refused(built, tmp_path, shape):
    npz = _copy(built, tmp_path / f"badshape-{type(shape).__name__}")

    def mutate(payload):
        payload["arrays"]["nhi_samples"]["shape"] = shape

    _rewrite_sidecar(npz, mutate)

    reasons = _refused(npz)
    assert any("arrays.nhi_samples.shape" in r for r in reasons)


def test_a_top_level_that_is_not_an_object_is_refused(built, tmp_path):
    npz = _copy(built, tmp_path / "toplevel")
    npz.with_suffix(".json").write_text('["not", "an", "object"]')

    reasons = _refused(npz)
    assert any("top level is a list" in r for r in reasons)


@pytest.mark.parametrize("section", ["arrays", "prior", "qmc"])
@pytest.mark.parametrize("value", [[], "text", 7], ids=["list", "string", "int"])
def test_a_section_with_the_wrong_type_is_refused(built, tmp_path, section, value):
    npz = _copy(built, tmp_path / f"badtype-{section}-{type(value).__name__}")

    def mutate(payload):
        payload[section] = value

    _rewrite_sidecar(npz, mutate)

    grid = load_sample_grid(path=npz)
    assert grid.usable_for_inference is False
    # Named, not merely refused.
    assert any(section in reason for reason in grid.unusable_because)


@pytest.mark.parametrize(
    "count",
    ["1200", 1200.5, None, True, [1200]],
    ids=["str", "float", "null", "bool", "list"],
)
def test_an_invalid_sample_count_is_refused(built, tmp_path, count):
    """``True`` is in the list on purpose: bool is an int in Python."""
    npz = _copy(built, tmp_path / f"badcount-{type(count).__name__}")

    def mutate(payload):
        payload["qmc"]["num_samples"] = count

    _rewrite_sidecar(npz, mutate)

    reasons = _refused(npz)
    assert any("qmc.num_samples" in r for r in reasons)


def test_a_required_array_missing_from_the_npz_fails_at_load(built, tmp_path):
    """A package-level validation error, and a clear one.

    ``offset_samples`` is required to build a coherent grid at all, so this
    fails when the arrays are read rather than at the integrity stage.
    """
    npz = _copy(built, tmp_path / "missingarray")
    with np.load(npz) as data:
        arrays = {k: data[k].copy() for k in data.files if k != "offset_samples"}
    np.savez_compressed(npz, **arrays)

    with pytest.raises(KeyError, match="offset_samples"):
        load_sample_grid(path=npz)


def test_an_optional_array_missing_from_the_npz_is_an_integrity_failure(
    built, tmp_path
):
    """``nhi_samples`` is optional to the dataclass but required by the sidecar.

    So its absence is a mismatch between the record and the file, which is
    exactly what the integrity check is for -- not a construction error.
    """
    npz = _copy(built, tmp_path / "no-nhi")
    with np.load(npz) as data:
        arrays = {k: data[k].copy() for k in data.files if k != "nhi_samples"}
    np.savez_compressed(npz, **arrays)

    grid = load_sample_grid(path=npz)
    assert grid.usable_for_inference is False
    assert any(
        "arrays.nhi_samples: absent from the file" in r for r in grid.unusable_because
    )


def test_no_malformed_sidecar_leaks_a_raw_exception(built, tmp_path):
    """The property behind all of the above, asserted directly.

    Whatever the sidecar contains, loading either raises a clear package-level
    error or returns a grid that reports itself unusable. It must never raise
    AttributeError or TypeError from inside the provenance reader.
    """
    payloads = [
        "[]",
        '"just a string"',
        "42",
        "null",
        '{"arrays": "text", "prior": 3, "qmc": []}',
        '{"sha256": 12345, "arrays": {"offset_samples": "x"}}',
        '{"arrays": {"log_nhi_samples": {"shape": "nope"}}}',
        '{"qmc": {"num_samples": {"n": 3}}}',
    ]
    for index, payload in enumerate(payloads):
        npz = _copy(built, tmp_path / f"fuzz{index}")
        npz.with_suffix(".json").write_text(payload)
        try:
            grid = load_sample_grid(path=npz)
        except (AttributeError, TypeError) as error:  # pragma: no cover
            raise AssertionError(
                f"payload {payload!r} leaked {type(error).__name__}: {error}"
            ) from error
        assert grid.usable_for_inference is False, payload
        assert grid.unusable_because, payload
