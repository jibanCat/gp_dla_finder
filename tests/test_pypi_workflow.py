"""Structural safety gates for the irreversible production-PyPI workflow."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "pypi.yml"


@pytest.fixture(scope="module")
def text() -> str:
    return WORKFLOW.read_text()


@pytest.fixture(scope="module")
def workflow() -> dict:
    yaml = pytest.importorskip("yaml")
    return yaml.safe_load(WORKFLOW.read_text())


def _triggers(workflow: dict) -> dict:
    return workflow.get("on", workflow.get(True))


def test_manual_only_with_all_identity_inputs(workflow, text):
    triggers = _triggers(workflow)
    assert set(triggers) == {"workflow_dispatch"}
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert set(inputs) == {
        "expected_version",
        "expected_commit",
        "testpypi_run_id",
        "confirm",
    }
    assert all(item["required"] is True for item in inputs.values())
    assert '"publish to pypi"' in text


def test_default_permissions_are_empty_and_only_publisher_has_oidc(workflow):
    assert workflow["permissions"] == {}
    holders = [
        name
        for name, job in workflow["jobs"].items()
        if (job.get("permissions") or {}).get("id-token") == "write"
    ]
    assert holders == ["publish-pypi"]
    assert workflow["jobs"]["stage"]["permissions"] == {
        "actions": "read",
        "contents": "read",
    }
    assert workflow["jobs"]["smoke-pypi"]["permissions"] == {}


def test_pending_publisher_identity_is_exact(workflow):
    publish = workflow["jobs"]["publish-pypi"]
    assert publish["environment"]["name"] == "pypi"
    assert WORKFLOW.name == "pypi.yml"


def test_oidc_job_never_checks_out_or_executes_project_code(workflow):
    steps = workflow["jobs"]["publish-pypi"]["steps"]
    uses = [step.get("uses", "") for step in steps]
    assert not any("checkout" in item for item in uses)
    assert any("download-artifact" in item for item in uses)
    assert any("pypa/gh-action-pypi-publish" in item for item in uses)
    for step in steps:
        run = step.get("run", "")
        assert "python" not in run
        assert "pip" not in run
        assert "build" not in run


def test_it_promotes_an_exact_successful_testpypi_run_without_rebuilding(text):
    for required in (
        '.path == ".github/workflows/release.yml"',
        '.event == "workflow_dispatch"',
        '.conclusion == "success"',
        '.head_branch == "main"',
        ".head_sha == $commit",
        "run-id: ${{ inputs.testpypi_run_id }}",
        "name: release-distributions",
        "github-token: ${{ github.token }}",
        "repository: ${{ github.repository }}",
    ):
        assert required in text
    assert "python -m build" not in text


def test_version_commit_tag_shape_and_both_hashes_are_gated(text):
    for required in (
        'ACTUAL_REF" = "refs/heads/main',
        'EXPECTED_COMMIT" = "$ACTUAL_COMMIT',
        'git cat-file -t "refs/tags/v${EXPECTED_VERSION}"',
        'tag_type" = "tag',
        "refs/tags/v${EXPECTED_VERSION}^{}",
        "check_release_version.py",
        "check_release_shape.py",
        "sha256sum --check SHA256SUMS.txt",
        "WHEEL_SHA256",
        "SDIST_SHA256",
    ):
        assert required in text
    assert "is_prerelease" in text
    assert "is_devrelease" in text


def test_testpypi_public_wheel_and_sdist_must_match_the_artifact(text):
    stage = text.split("  stage:", 1)[1].split("  publish-pypi:", 1)[0]
    assert "https://test.pypi.org/pypi/gp-dla-finder/{version}/json" in stage
    assert 'item["digests"]["sha256"]' in stage
    assert "published != wanted" in stage
    assert 'os.environ["WHEEL"]' in stage
    assert 'os.environ["SDIST"]' in stage


_USES = re.compile(r"uses:\s*([^\s@]+)@([^\s#]+)")


def test_every_action_is_immutable_and_documented(text):
    for line in text.splitlines():
        if "uses:" not in line or "@" not in line:
            continue
        match = _USES.search(line)
        assert match and re.fullmatch(r"[0-9a-f]{40}", match.group(2)), line
        assert re.search(r"#\s*v[\d.]+", line), line


def test_no_secret_or_api_token_and_no_test_index_in_production_job(text):
    assert "secrets." not in text
    assert "api-token" not in text
    publish = text.split("  publish-pypi:", 1)[1].split("  smoke-pypi:", 1)[0]
    assert "test.pypi.org" not in publish
    assert "repository-url:" not in publish


def test_smoke_downloads_wheel_and_sdist_and_compares_exact_hashes(text):
    smoke = text.split("  smoke-pypi:", 1)[1]
    assert "https://pypi.org/pypi/gp-dla-finder/${version}/json" not in smoke
    assert "https://pypi.org/pypi/gp-dla-finder/{version}/json" in smoke
    assert "set(files) != wanted" in smoke
    assert "${WHEEL_SHA256}" in smoke
    assert "${SDIST_SHA256}" in smoke
    assert 'assert __version__ == os.environ["VERSION"]' in smoke
