"""The release workflow's safety properties, asserted rather than reviewed once.

A publishing workflow is the one place in this repository where a mistake is
irreversible: an upload cannot be recalled, and a name cannot be reused. So the
properties that make it safe are pinned here, where a later edit that removes
one fails a test instead of being noticed by whoever happens to read the diff.

The properties:

* it runs **only** when a human dispatches it, never on a push or a tag;
* it publishes **only** to TestPyPI -- there is no production path in the file;
* the job holding the OIDC credential does not check out this repository and
  does not run its build code;
* no job requests more permission than it needs, and the default is none;
* third-party actions are pinned to immutable commit SHAs, so a retagged
  upstream release cannot change what runs; and
* no API token or password appears anywhere.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"


@pytest.fixture(scope="module")
def workflow() -> dict:
    # importorskip INSIDE the fixture, never at module level. A module-level
    # skip fires during collection, before marker filtering can exclude the
    # module -- so a job selecting `-m needs_reference` still records a skip
    # for a file it never wanted, and the canonical-parity skip contract
    # rightly fails. That is how this landed the first time.
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text())


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text()


def _triggers(workflow: dict) -> dict:
    # PyYAML parses the bare key `on` as the boolean True.
    return workflow.get("on", workflow.get(True))


# --- when it can run ---------------------------------------------------------


def test_it_is_manual_only(workflow):
    """No push, no tag, no schedule. A release is a decision, not an event."""
    triggers = _triggers(workflow)
    assert set(triggers) == {"workflow_dispatch"}, triggers


def test_it_requires_a_typed_confirmation(workflow, text):
    """A dispatch is one click; this makes it a deliberate sentence."""
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert "confirm" in inputs
    assert inputs["confirm"]["required"] is True
    assert "publish to testpypi" in text
    # And the check actually gates the build.
    # Compared through the environment, not interpolated into the shell.
    assert '"$CONFIRM" != "publish to testpypi"' in text


# --- where it can publish ----------------------------------------------------


def test_it_publishes_only_to_testpypi(text):
    assert "test.pypi.org/legacy/" in text
    # The production endpoint must not appear at all -- not commented out, not
    # behind a condition. A reachable production path is one misclick from an
    # irreversible upload.
    assert "upload.pypi.org" not in text
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "repository-url: https://pypi.org" not in stripped


def test_the_environment_matches_the_pending_publisher(workflow):
    """These four fields are what TestPyPI is asked to trust.

    They must match the pending publisher exactly, so they are pinned here and
    quoted in the release note the PI works from.
    """
    publish = workflow["jobs"]["publish-testpypi"]
    assert publish["environment"]["name"] == "testpypi"
    assert WORKFLOW.name == "release.yml"


# --- who holds the credential ------------------------------------------------


def test_the_default_permission_is_nothing(workflow):
    assert workflow["permissions"] == {}


def test_only_the_publish_job_can_request_an_oidc_token(workflow):
    holders = [
        name
        for name, job in workflow["jobs"].items()
        if (job.get("permissions") or {}).get("id-token") == "write"
    ]
    assert holders == ["publish-testpypi"], holders


def test_the_build_job_holds_no_publishing_credential(workflow):
    build = workflow["jobs"]["build"]
    assert build["permissions"] == {"contents": "read"}
    assert "id-token" not in build["permissions"]


def test_the_publish_job_does_not_check_out_or_build(workflow):
    """The OIDC job runs no code from THIS repository.

    Stated precisely, because "runs almost nothing" was too loose: the job
    necessarily executes the pinned artifact action, two checksum commands and
    the pinned PyPA publishing action. What it does not do is check out this
    repository or run its source or package code, so the credential is never
    live in a job executing project code.
    """
    steps = workflow["jobs"]["publish-testpypi"]["steps"]
    uses = [step.get("uses", "") for step in steps]
    assert not any("actions/checkout" in u for u in uses), uses
    assert any("download-artifact" in u for u in uses), uses
    # And it runs no build commands.
    for step in steps:
        run = step.get("run", "")
        assert "python -m build" not in run
        assert "pip install -e" not in run


def test_the_smoke_job_has_no_credential(workflow):
    assert workflow["jobs"]["smoke-testpypi"]["permissions"] == {}


# --- what it trusts ----------------------------------------------------------


_USES = re.compile(r"uses:\s*([^\s@]+)@([^\s#]+)")


def test_every_action_is_pinned_to_a_commit_sha(text):
    """A tag can be moved; a commit cannot.

    Pinning by SHA means a retagged upstream release cannot silently change
    what runs in the job that holds the publishing credential.
    """
    unpinned = [
        f"{name}@{ref}"
        for name, ref in _USES.findall(text)
        if not re.fullmatch(r"[0-9a-f]{40}", ref)
    ]
    assert not unpinned, f"actions pinned to a mutable ref: {unpinned}"


def test_each_pin_records_the_version_it_corresponds_to(text):
    """A bare SHA is unreadable; the trailing comment says what it is."""
    for line in text.splitlines():
        if "uses:" in line and "@" in line:
            assert re.search(r"#\s*v[\d.]+", line), (
                f"pin without a version: {line.strip()}"
            )


def test_the_publish_action_is_the_pypa_one(text):
    assert "pypa/gh-action-pypi-publish@" in text


# --- what must never appear --------------------------------------------------


def test_no_token_or_password_is_used(text):
    """Trusted Publishing exists so that no long-lived credential exists."""
    lowered = text.lower()
    for forbidden in ("password:", "api_token", "pypi_token", "twine_password"):
        assert forbidden not in lowered, forbidden
    assert "secrets." not in text, "the release path should need no repository secret"


def test_the_wheel_is_built_without_the_compiled_extension(text):
    """Otherwise the runner ships a platform wheel whenever libcerf is present."""
    assert "GP_DLA_FINDER_NO_COMPILE=1 python -m build --wheel" in text


def test_the_artifacts_are_hashed_and_the_hashes_are_checked(text):
    """The manifest is what makes "the public files are the tested files"
    checkable rather than asserted."""
    assert "sha256sum *.whl *.tar.gz | tee SHA256SUMS.txt" in text
    assert "sha256sum --check SHA256SUMS.txt" in text


# --- publishing the right version, and testing that one ----------------------


def test_the_version_must_be_stated_explicitly(workflow):
    """A correct confirmation phrase must not be able to publish the wrong thing."""
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert "expected_version" in inputs
    assert inputs["expected_version"]["required"] is True


def test_the_version_gate_runs_before_anything_is_built(text):
    gate = text.index("check_release_version.py")
    build = text.index("python -m build --wheel")
    assert gate < build, "the version is checked after building"


def test_the_built_filenames_are_checked_against_the_version(text):
    """Metadata agreeing is not the same as the bytes agreeing.

    A stale dist/ is how correct metadata and the wrong files end up in one
    upload, so the built names are checked too.
    """
    assert '--expected "$EXPECTED_VERSION" --dist dist/' in text


def test_the_verified_version_is_carried_to_the_smoke_job(workflow):
    """Not "whatever is newest on TestPyPI".

    On a re-run, or alongside another candidate, the newest version there is
    not necessarily the one this run just published.
    """
    assert workflow["jobs"]["build"]["outputs"]["version"]
    smoke = workflow["jobs"]["smoke-testpypi"]
    assert "build" in smoke["needs"]
    assert smoke["env"]["VERSION"] == "${{ needs.build.outputs.version }}"


def test_the_smoke_job_installs_the_exact_version(text):
    """Checked on the pip command, not on an error message.

    An earlier version of this test looked for the pinned spec anywhere in the
    file. It passed a mutation that unpinned the actual download, because the
    same string also appears in the "never appeared on TestPyPI" error text --
    the assertion was reading prose rather than reading the command.
    """
    # A pip invocation may be split across continuation lines, so join first
    # and then look at each command as a whole.
    joined = text.replace("\\\n", " ")
    commands = [
        " ".join(line.split())
        for line in joined.splitlines()
        if ("pip download" in line or "pip install" in line)
    ]
    assert commands, "the smoke job runs no pip command"

    resolving = [
        command
        for command in commands
        if "gp_dla_finder" in command and ".whl" not in command
    ]
    assert resolving, "nothing resolves the package from an index"
    for command in resolving:
        assert "gp_dla_finder==${VERSION}" in command, (
            f"unpinned package spec in: {command}"
        )

    # And the installed package is asserted to be that version at run time.
    assert "__version__ == expected" in text


def test_the_package_provably_comes_from_testpypi(text):
    """One resolver call across two indexes cannot say which index won.

    So the wheel is downloaded from TestPyPI with dependencies disabled, its
    filename is checked, and only then is it installed -- letting its
    dependencies come from the normal index.
    """
    assert "--index-url https://test.pypi.org/simple/" in text
    assert "--no-deps --only-binary :all:" in text
    assert "gp_dla_finder-${VERSION}-py3-none-any.whl" in text
    # The ambiguous two-index install must be gone.
    assert "--extra-index-url" not in text


def test_the_smoke_job_still_retries_for_propagation(text):
    """A new upload is not instantly visible on the index."""
    assert "not on the index yet" in text


def test_a_publish_is_never_cancelled_midway(workflow):
    assert workflow["concurrency"]["cancel-in-progress"] is False


# --- the version gate itself, as a unit --------------------------------------
#
# Driven directly rather than through the workflow, because the interesting
# cases are the refusals and a workflow cannot be run locally.


def _gate():
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from check_release_version import check

    return check


def test_a_development_version_is_refused():
    """The version currently in pyproject.toml, so this is not hypothetical.

    A .dev number moves whenever someone feels like it, and an index filename
    is permanent.
    """
    (problem,) = _gate()("0.1.0.dev0", "0.1.0.dev0")
    assert "development version" in problem


def test_an_empty_expected_version_is_refused():
    (problem,) = _gate()("", "0.1.0rc1")
    assert "no expected version" in problem


def test_a_mismatch_between_expected_and_declared_is_refused():
    (problem,) = _gate()("0.1.0rc1", "0.1.0rc2")
    assert "does not match pyproject.toml" in problem


def test_a_local_version_is_refused():
    problems = _gate()("0.1.0rc1+local", "0.1.0rc1+local")
    assert any("local version" in p for p in problems)


def test_a_release_candidate_is_accepted():
    assert _gate()("0.1.0rc1", "0.1.0rc1") == []


@pytest.mark.parametrize("version", ["0.1.0", "0.1.0rc1", "0.2.0a1", "1.0.0b2"])
def test_real_release_versions_are_accepted(version):
    assert _gate()(version, version) == []


def test_a_built_filename_that_disagrees_is_refused(tmp_path):
    """Correct metadata, wrong bytes -- which a stale dist/ produces."""
    (tmp_path / "gp_dla_finder-0.1.0rc1-py3-none-any.whl").write_bytes(b"")
    (tmp_path / "gp_dla_finder-0.1.0.dev0.tar.gz").write_bytes(b"")

    problems = _gate()("0.1.0rc1", "0.1.0rc1", tmp_path)
    assert any("0.1.0.dev0" in p and "expected '0.1.0rc1'" in p for p in problems)


def test_matching_built_filenames_are_accepted(tmp_path):
    (tmp_path / "gp_dla_finder-0.1.0rc1-py3-none-any.whl").write_bytes(b"")
    (tmp_path / "gp_dla_finder-0.1.0rc1.tar.gz").write_bytes(b"")
    assert _gate()("0.1.0rc1", "0.1.0rc1", tmp_path) == []


def test_the_gate_reads_a_projects_declared_version(tmp_path):
    """It must read pyproject.toml rather than carry a constant.

    Checked against a temporary project, deliberately. Asserting the live
    version here would make the intended bump to 0.1.0rc1 fail this test --
    a guard that fires on the change it exists to permit.
    """
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from check_release_version import project_version

    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "probe"\nversion = "9.9.9rc7"\n'
    )
    assert project_version(tmp_path) == "9.9.9rc7"


# --- the source gate: main, and the commit someone named ---------------------


def test_the_commit_must_be_stated_explicitly(workflow):
    """Three questions, three answers.

    The phrase says publication was intended. The version says which release
    number. The commit says which reviewed source. A confirmation phrase that
    also settled the source would collapse two decisions into one.
    """
    inputs = _triggers(workflow)["workflow_dispatch"]["inputs"]
    assert "expected_commit" in inputs
    assert inputs["expected_commit"]["required"] is True


def test_the_source_gate_refuses_a_branch_other_than_main(text):
    assert 'if [ "$ACTUAL_REF" != "refs/heads/main" ]' in text
    assert "releases are built from main only" in text


def test_the_source_gate_requires_a_full_sha(text):
    """A short SHA is ambiguous and a symbolic ref is not a commit."""
    assert "must be a full 40-character SHA" in text
    # Forty bracketed hex classes, one per character.
    assert text.count("[0-9a-f]") == 40


def test_the_source_gate_compares_against_the_dispatched_commit(text):
    assert 'if [ "$EXPECTED_COMMIT" != "$ACTUAL_COMMIT" ]' in text


def test_the_source_gate_runs_before_the_build(text):
    gate = text.index("releases are built from main only")
    build = text.index("python -m build --wheel")
    assert gate < build


def test_the_verified_commit_is_recorded_as_an_output(workflow):
    """Release evidence carries the commit beside the version and the digests."""
    assert workflow["jobs"]["build"]["outputs"]["commit"]


# --- dispatch input must never reach shell text ------------------------------


def test_no_dispatch_input_is_interpolated_into_a_shell_command(text):
    """`${{ inputs.x }}` in a `run:` block is substituted before the shell runs.

    An input containing a quote or a semicolon then becomes script rather than
    data. Mapping inputs through the environment and quoting "$VARIABLES" keeps
    them as values.
    """
    offenders = []
    in_run = False
    run_indent = 0
    for number, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if stripped.startswith("run: |") or stripped == "run: >":
            in_run = True
            run_indent = len(raw) - len(raw.lstrip())
            continue
        if in_run:
            indent = len(raw) - len(raw.lstrip())
            if stripped and indent <= run_indent:
                in_run = False
            elif "${{" in raw and "inputs." in raw:
                offenders.append(f"{number}: {stripped}")
    assert not offenders, "dispatch inputs interpolated into shell text:\n" + "\n".join(
        offenders
    )


def test_every_dispatch_input_is_mapped_through_the_environment(workflow):
    env = workflow["jobs"]["build"]["env"]
    mapped = {v for v in env.values() if isinstance(v, str)}
    for name in ("confirm", "expected_version", "expected_commit"):
        assert any(f"inputs.{name}" in value for value in mapped), name


def test_downstream_jobs_receive_validated_values_not_raw_input(workflow):
    """The smoke job reads the build job's output, not the dispatch input."""
    smoke = workflow["jobs"]["smoke-testpypi"]
    assert smoke["env"]["VERSION"] == "${{ needs.build.outputs.version }}"
    assert "inputs." not in str(smoke["env"])


# --- dependency resolution is stated, not inherited --------------------------


def test_the_install_is_isolated_from_runner_index_configuration(text):
    """pip.conf or a PIP_INDEX_URL on the runner could otherwise supply a
    dependency, and nothing in the log would say so."""
    joined = text.replace("\\\n", " ")
    commands = [
        " ".join(line.split())
        for line in joined.splitlines()
        if ("pip download" in line or "pip install" in line)
        # Only the commands that reach an index for the package under test.
        # Bootstrapping pip itself, and installing this checkout to run the
        # artifact checks, are not what this guards.
        and ("test.pypi.org" in line or "testpypi-wheel" in line)
    ]
    assert commands, "the smoke job runs no index-facing pip command"
    for command in commands:
        assert "--isolated" in command, f"not isolated: {command}"


def test_dependencies_come_from_production_pypi_explicitly(text):
    assert "--index-url https://pypi.org/simple/" in text


# --- the version gate, canonical forms ---------------------------------------


@pytest.mark.parametrize(
    "version", ["v0.1.0", "0.1.0-rc1", "abc", "1.0.0.rc1", "0.1.0rc", ""]
)
def test_a_non_canonical_version_is_refused(version):
    assert _gate()(version, version) != []


@pytest.mark.parametrize(
    # "0.1" is included deliberately: it is a perfectly canonical PEP 440
    # version, and an earlier draft of this test wrongly listed it as invalid.
    "version",
    ["0.1.0", "0.1.0rc1", "1.2.0b3", "0.1.0.post1", "0.1"],
)
def test_a_prerelease_or_final_release_is_accepted(version):
    """The workflow must be able to validate the final 0.1.0 too, not only rc1."""
    assert _gate()(version, version) == []


# --- whitespace, normalisation, and the validated value ----------------------


@pytest.mark.parametrize(
    "version",
    [" 0.1.0rc1", "0.1.0rc1 ", " 0.1.0rc1 ", "0.1.0 rc1", "0.1.0rc1\n", "\t0.1.0rc1"],
    ids=["leading", "trailing", "both", "embedded", "newline", "tab"],
)
def test_whitespace_in_the_version_is_refused_not_stripped(version):
    """Refused, so the validated value and the typed value are the same string.

    Stripping would mean the gate validated one string while a later step used
    another -- and a trailing newline would additionally let a second line into
    $GITHUB_OUTPUT.
    """
    problems = _gate()(version, version)
    assert problems, f"{version!r} was accepted"
    assert "whitespace" in problems[0]


@pytest.mark.parametrize(
    "version, means",
    [
        ("v0.1.0", "0.1.0"),
        ("0.1.0-rc1", "0.1.0rc1"),
        ("0.1.0.rc1", "0.1.0rc1"),
        ("0.01.0", "0.1.0"),
        ("1.0.0.0", "1.0.0.0"),
    ],
    ids=["v-prefix", "dash", "dot-rc", "leading-zero", "trailing-zero"],
)
def test_a_valid_but_non_normalised_spelling_is_refused(version, means):
    """These parse, and name the same release through different spellings.

    A release should have one name. `0.01.0` and `0.1.0` are the same version
    to any index, so accepting both as canonical would let two strings denote
    one file.
    """
    problems = _gate()(version, version)
    if version == means:
        pytest.skip(f"{version} is already normalised")
    assert problems, f"{version!r} was accepted"
    assert any("normalised spelling" in p for p in problems)
    assert any(means in p for p in problems)


@pytest.mark.parametrize(
    "version", ["0.1", "0.1.0", "0.1.0rc1", "1.2.0b3", "0.1.0.post1", "2!1.0.0"]
)
def test_canonical_release_versions_are_accepted(version):
    """`0.1` included deliberately: it is canonical PEP 440.

    An earlier diagnostic named it as an example of an invalid version while
    the tests accepted it. The message and the policy now agree.
    """
    assert _gate()(version, version) == []


def test_the_invalid_version_message_does_not_contradict_the_policy():
    """The diagnostic must not cite something the gate accepts."""
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from check_release_version import check

    (message,) = check("!!!", "!!!")
    for accepted in ("'0.1'", '"0.1"'):
        assert accepted not in message, (
            f"the message offers {accepted} as invalid, but the gate accepts it"
        )


def test_the_workflow_exports_the_validated_value_not_its_input(text):
    """The invariant: downstream receives what was checked.

    Echoing $EXPECTED_VERSION would export the raw dispatch input, so any
    difference between the typed and validated values would propagate
    unchecked.
    """
    assert "--github-output" in text
    assert 'echo "version=$EXPECTED_VERSION"' not in text
    assert "version=${{ inputs.expected_version }}" not in text


def test_nothing_is_exported_when_validation_fails(tmp_path, monkeypatch):
    """A refusal must not leave a version behind for a later step to read."""
    import subprocess
    import sys

    out = tmp_path / "gh-output"
    out.write_text("")
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "check_release_version.py"),
            "--expected",
            "0.1.0.dev0",
            "--github-output",
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "GITHUB_OUTPUT": str(out)},
        timeout=120,
    )
    assert completed.returncode == 1
    assert out.read_text() == "", "a refused version was exported anyway"


# --- the adversarial pass's two findings -------------------------------------


def test_the_build_directory_is_emptied_first(text):
    """ "Should be empty" and "is empty" are different statements.

    A runner checkout is fresh, so nothing should be in dist/. Emptying it
    makes the property hold by construction rather than by how the job was
    reached.
    """
    clean = text.index("rm -rf dist/")
    build = text.index("python -m build --wheel")
    assert clean < build


def test_exactly_one_distribution_of_each_kind_is_required(text):
    """A stray file must fail rather than be uploaded."""
    shape = (ROOT / "tools" / "check_release_shape.py").read_text()
    assert "expected exactly one wheel" in shape
    assert "expected exactly one sdist" in shape
    assert "check_release_shape.py dist/" in text


def test_the_testpypi_download_bypasses_the_pip_cache(text):
    """--isolated ignores configuration but not the HTTP cache.

    A wheel cached by an earlier run could otherwise be served without
    TestPyPI being contacted, making "it came from TestPyPI in this run" an
    assumption rather than something demonstrated.
    """
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        if "pip download" in line and "test.pypi.org" in line:
            assert "--no-cache-dir" in line, line
            break
    else:  # pragma: no cover
        pytest.fail("no TestPyPI download command found")


# --- publication paths that must not exist -----------------------------------


def test_there_is_no_reusable_workflow_or_scheduled_entry_point(workflow):
    """workflow_call, push, schedule and repository_dispatch would each be a
    way to publish without a person deciding to."""
    triggers = _triggers(workflow)
    assert set(triggers) == {"workflow_dispatch"}
    for forbidden in ("workflow_call", "push", "schedule", "repository_dispatch"):
        assert forbidden not in triggers


def test_no_job_uses_a_matrix(workflow):
    """A matrix would multiply the upload attempts."""
    for name, job in workflow["jobs"].items():
        assert "strategy" not in job, f"{name} has a strategy/matrix"


def test_a_concurrent_dispatch_queues_rather_than_racing(workflow):
    concurrency = workflow["concurrency"]
    assert concurrency["cancel-in-progress"] is False
    assert "release-" in concurrency["group"]


# --- the release bootstrap must exist, and must come first -------------------
#
# The version gate imports `packaging`. It used to run before anything was
# installed, so on a clean interpreter it died with ModuleNotFoundError before
# it could refuse anything -- and appeared to work only where a developer
# machine happened to have packaging already. Two tests: one on the declared
# order, one that actually runs the checker somewhere clean.


def _bootstrap_specs(text: str) -> list[str]:
    """The requirement specs the workflow's bootstrap step installs.

    Read out of the workflow so a test cannot drift from what the release job
    actually does.
    """
    joined = text.replace("\\\n", " ")
    for line in joined.splitlines():
        if "pip install" in line and "packaging" in line:
            return re.findall(r"'([^']+)'", line)
    return []


def test_the_bootstrap_specs_are_readable_from_the_workflow():
    specs = _bootstrap_specs(WORKFLOW.read_text())
    assert any(spec.startswith("packaging") for spec in specs), specs
    assert any(spec.startswith("tomli") for spec in specs), specs


def test_the_bootstrap_covers_the_toml_reader_on_older_interpreters(text):
    """tomllib is stdlib only from 3.11, and the gate reads pyproject.toml.

    This job pins 3.12 so it would not bite today -- but "would not bite
    today" is exactly how the packaging gap got in.
    """
    assert 'tomli>=2.0,<3; python_version<"3.11"' in text


def test_the_bootstrap_installs_packaging_explicitly(text):
    """Stated directly, not relied on as a transitive of `build`.

    A dependency that arrives by accident is one upstream change away from not
    arriving.
    """
    assert "packaging>=23,<27" in text


def test_the_bootstrap_bounds_every_tool_it_installs(text):
    """Matching the ceiling policy the package uses for its own dependencies."""
    for spec in ("packaging>=23,<27", "build>=1.0,<2", "twine>=5,<7"):
        assert spec in text, f"missing or unbounded: {spec}"


def test_the_bootstrap_runs_after_setup_python_and_before_the_gate(workflow):
    """The order the whole correction is about."""
    names = [
        (step.get("name") or step.get("uses") or "")
        for step in workflow["jobs"]["build"]["steps"]
    ]

    def index_of(fragment):
        for position, name in enumerate(names):
            if fragment in name:
                return position
        raise AssertionError(f"no step matching {fragment!r} in {names}")

    setup = index_of("actions/setup-python")
    bootstrap = index_of("Install the release bootstrap")
    gate = index_of("Refuse a version nobody asked for")
    build = index_of("Build the universal wheel")

    assert setup < bootstrap < gate < build, (
        f"expected setup-python -> bootstrap -> version gate -> build, got "
        f"{setup} -> {bootstrap} -> {gate} -> {build}"
    )


def test_the_source_gates_still_precede_the_checkout(workflow):
    """Refusing a bad source should not require fetching it first."""
    names = [
        (step.get("name") or step.get("uses") or "")
        for step in workflow["jobs"]["build"]["steps"]
    ]
    checkout = next(i for i, n in enumerate(names) if "actions/checkout" in n)
    confirm = next(i for i, n in enumerate(names) if "Refuse an unconfirmed run" in n)
    source = next(i for i, n in enumerate(names) if "Refuse a source" in n)
    assert confirm < checkout and source < checkout


#: Modules the release bootstrap must make importable before the version gate
#: can run. ``tomli`` only below 3.11, where ``tomllib`` is not yet stdlib.
def _modules_the_gate_needs() -> tuple[str, ...]:
    import sys

    if sys.version_info < (3, 11):
        return ("packaging", "tomli")
    return ("packaging",)


def _module_present(python, module: str) -> bool:
    """Whether ``module`` is importable by ``python``, by return code alone.

    Deliberately not a substring check on output. An earlier version asserted
    that the module name appeared in the failure text, which a traceback
    satisfies whenever any directory on the path happens to share that name --
    and then reports success while the real failure was something else
    entirely. An exit status carries no filenames, so it cannot be fooled that
    way. The next test demonstrates both halves synthetically.
    """
    import subprocess

    completed = subprocess.run(
        [
            str(python),
            "-c",
            "import importlib.util as u, sys; "
            f"sys.exit(0 if u.find_spec({module!r}) else 1)",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert completed.returncode in (0, 1), (
        f"probe for {module!r} failed unexpectedly: {completed.stderr}"
    )
    return completed.returncode == 0


@pytest.mark.slow
def test_the_checker_starts_on_a_clean_interpreter(tmp_path):
    """Run the gate where nothing is installed but the bootstrap.

    This is the test that caught the ordering defect: a clean environment on
    the release job's own Python has no ``packaging``, so the checker cannot
    reach its own logic until the bootstrap installs it.

    Every step is a return code or an exact exception, never a substring of a
    traceback.
    """
    import subprocess
    import sys
    import venv

    env_dir = tmp_path / "clean"
    venv.EnvBuilder(with_pip=True).create(env_dir)
    python = env_dir / "bin" / "python"
    if not python.exists():  # pragma: no cover - Windows layout
        python = env_dir / "Scripts" / "python.exe"

    needed = _modules_the_gate_needs()

    # 1. Before the bootstrap, every module the gate needs is genuinely absent.
    for module in needed:
        assert not _module_present(python, module), (
            f"{module!r} is already importable in a clean environment; this "
            "test cannot establish what the bootstrap provides"
        )

    checker = ROOT / "tools" / "check_release_version.py"

    # 2. And the checker cannot run, failing on a module it needs -- matched by
    #    the exception's module name, not by searching the message.
    bare = subprocess.run(
        [
            str(python),
            "-c",
            "import runpy, sys\n"
            "sys.argv = ['check', '--expected', '0.1.0rc1']\n"
            "try:\n"
            f"    runpy.run_path({str(checker)!r}, run_name='__main__')\n"
            "except ModuleNotFoundError as error:\n"
            "    print(error.name)\n"
            "    sys.exit(42)\n"
            "except SystemExit:\n"
            "    raise\n"
            "sys.exit(0)",
        ],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert bare.returncode == 42, (
        f"expected a ModuleNotFoundError, got exit {bare.returncode}: "
        f"{bare.stdout}{bare.stderr}"
    )
    missing = bare.stdout.strip().splitlines()[-1]
    assert missing in needed, (
        f"the checker failed on {missing!r}, which is not one of the modules "
        f"the bootstrap provides ({needed})"
    )

    # 3. Install exactly what the committed workflow installs.
    specs = _bootstrap_specs(WORKFLOW.read_text())
    assert specs, "no bootstrap specs found in the workflow"
    subprocess.run(
        [str(python), "-m", "pip", "install", "-q", *specs],
        check=True,
        timeout=900,
    )

    # 4. Now every needed module imports.
    for module in needed:
        assert _module_present(python, module), (
            f"the workflow's bootstrap does not provide {module!r} on "
            f"Python {sys.version_info.major}.{sys.version_info.minor}"
        )

    # 5. And the real checker works against a real pyproject.toml.
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        '[project]\nname = "probe"\nversion = "0.1.0rc1"\n'
    )

    ok = subprocess.run(
        [str(python), str(checker), "--expected", "0.1.0rc1", "--root", str(project)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert ok.returncode == 0, ok.stdout + ok.stderr

    refused = subprocess.run(
        [str(python), str(checker), "--expected", "v0.1.0rc1", "--root", str(project)],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert refused.returncode == 1
    assert "normalised spelling" in refused.stdout


def test_a_substring_probe_is_fooled_by_a_path_and_an_exit_code_is_not(tmp_path):
    """The regression, reproduced from materials this test creates itself.

    Nothing here depends on where the repository happens to live. The
    directory carrying the marker word is built under ``tmp_path``, so the
    demonstration behaves identically wherever the checkout is -- a developer
    machine, a continuous-integration runner, anywhere. That is the whole
    point: the previous version asserted a property of one particular
    developer's filesystem and duly failed on CI, whose checkout directory is
    named after the repository instead.

    No absolute path appears in this file, deliberately. ``tests/`` ships in
    the source distribution, and the artifact scan refuses a build-machine
    path wherever it finds one -- including in a docstring explaining why
    build-machine paths are a problem.
    """
    import subprocess
    import sys

    marker = "packaging"
    absent = "a_module_that_does_not_exist_xyz"

    # A script under a directory whose NAME contains the marker, failing on a
    # module that has nothing to do with the marker.
    workdir = tmp_path / f"{marker}_checkout" / "tools"
    workdir.mkdir(parents=True)
    script = workdir / "probe.py"
    script.write_text(f"import {absent}\n")

    completed = subprocess.run(
        [sys.executable, str(script)], capture_output=True, text=True, timeout=120
    )
    combined = completed.stdout + completed.stderr

    # The failure is about `absent`, and says so.
    assert completed.returncode != 0
    assert absent in combined

    # Yet the OLD style of assertion -- "the marker appears in the output" --
    # is satisfied, purely because the traceback prints the script's path.
    assert marker in combined, "expected the path to put the marker in the output"
    assert marker not in absent, "the marker must be unrelated to the real failure"

    # So a substring check would have reported that `marker` was the missing
    # module. The exit-status probe answers correctly instead: it reports a
    # standard-library module present and a nonexistent one absent, with no
    # output inspected at all.
    assert _module_present(sys.executable, "json") is True
    assert _module_present(sys.executable, absent) is False


# --- the diagnostics must describe the rule they enforce ---------------------


@pytest.mark.parametrize("version", ["abc", "1.0.0!!", "..", "0.1.0~rc1"])
def test_an_unparseable_version_says_so(version):
    (problem,) = _gate()(version, version)
    assert "cannot be parsed" in problem


@pytest.mark.parametrize(
    # "0.1.0rc" is here, not in the unparseable list: packaging fills the
    # implicit zero and reads it as "0.1.0rc0". A third case where my
    # expectation about a version form was wrong and the parser was right,
    # which is the argument for using a parser rather than intuition.
    "version",
    ["v0.1.0", "0.1.0-rc1", "0.01.0", "0.1.0.rc1", "0.1.0rc"],
)
def test_a_parseable_but_non_normalised_version_says_that_instead(version):
    """These ARE PEP 440 versions. Saying otherwise would be false.

    packaging understands every one of them; this policy refuses them for
    their spelling, and the message has to say which of the two it is.
    """
    problems = _gate()(version, version)
    assert any("valid PEP 440 version" in p for p in problems)
    assert not any("cannot be parsed" in p for p in problems)


def test_each_diagnostic_only_cites_examples_that_follow_its_own_rule():
    """The examples in a message must obey the rule the message states."""
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    from check_release_version import _parse, check

    (unparseable,) = check("!!!", "!!!")
    quoted = re.findall(r"'([^']+)'", unparseable)
    # Every example offered as a valid form must really parse and be normalised.
    for example in quoted:
        if example == "!!!":
            continue
        parsed = _parse(example)
        assert parsed is not None, f"{example!r} is offered as valid but does not parse"
        assert str(parsed) == example, (
            f"{example!r} is offered as valid but is not normalised"
        )
