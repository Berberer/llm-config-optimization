"""Unit tests for llama.cpp's model-argument resolution.

``model_request`` decides whether the ``--model`` argument names a local
checkpoint or a Hugging Face repository, and for a repository it picks the exact
GGUF file to fetch. The observable contract is the neutral
:class:`~llm_config_optimization.inference.types.ModelRequest` it returns and the
:class:`~llm_config_optimization.inference.types.ExperimentError` it raises for an
argument that cannot be resolved unambiguously. A fake repository-file lister
stands in for the Hugging Face API, so the repository cases are covered without
the network; the local cases use real temporary files.
"""

from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from llm_config_optimization.inference import ExperimentError, HubCheckpoint, LocalCheckpoint
from llm_config_optimization.inference._llamacpp import model_request

REPO = "bartowski/Llama-3.2-1B-Instruct-GGUF"
"""The repository id every fake lister is asked about."""

_QUANTS = ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Llama-3.2-1B-Instruct-Q8_0.gguf")
"""The GGUF files of a typical multi-quant repository."""

_NON_GGUF = ("config.json", "README.md", "tokenizer.json")
"""Files a checkpoint repository holds besides its GGUF quants."""


def _repo(files: Sequence[str]) -> Callable[[str], Sequence[str]]:
    """Build a lister that serves ``files`` and pins the repository id."""

    def list_repo_files(repo_id: str) -> Sequence[str]:
        assert repo_id == REPO
        return list(files)

    return list_repo_files


def test_an_existing_file_is_a_local_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    checkpoint.write_bytes(b"GGUF")

    request = model_request(str(checkpoint), list_repo_files=_repo(()))

    assert request == LocalCheckpoint(path=checkpoint.resolve())


def test_a_local_checkpoint_needs_no_repository_lookup(tmp_path: Path) -> None:
    """The fake lister asserts it is never called for a local model."""
    checkpoint = tmp_path / "model.gguf"
    checkpoint.write_bytes(b"GGUF")

    def never_called(repo_id: str) -> Sequence[str]:
        raise AssertionError(f"unexpected repository lookup for {repo_id}")

    assert isinstance(model_request(str(checkpoint), list_repo_files=never_called), LocalCheckpoint)


def test_an_existing_directory_is_rejected_as_not_gguf(tmp_path: Path) -> None:
    """llama.cpp serves one file, so a snapshot directory cannot be a checkpoint."""
    with pytest.raises(ExperimentError, match="not a GGUF file"):
        model_request(str(tmp_path), list_repo_files=_repo(()))


def test_a_missing_path_is_resolved_as_a_repository() -> None:
    """A path that does not exist is a repository id, exactly like a Hub id."""
    request = model_request(REPO, list_repo_files=_repo(("model-Q8_0.gguf",)))

    assert request == HubCheckpoint(repo_id=REPO, filename="model-Q8_0.gguf")


def test_a_repository_with_one_gguf_needs_no_selector() -> None:
    files = [*_NON_GGUF, "Llama-3.2-1B-Instruct-Q8_0.gguf"]

    request = model_request(REPO, list_repo_files=_repo(files))

    assert request == HubCheckpoint(repo_id=REPO, filename="Llama-3.2-1B-Instruct-Q8_0.gguf")


def test_a_repository_with_several_gguf_files_requires_a_selector() -> None:
    with pytest.raises(ExperimentError, match="Q4_K_M") as error:
        model_request(REPO, list_repo_files=_repo(_QUANTS))

    assert "Q8_0" in str(error.value)
    assert f"{REPO}:QUANT" in str(error.value)


def test_a_repository_without_gguf_files_is_rejected() -> None:
    with pytest.raises(ExperimentError, match="no GGUF files"):
        model_request(REPO, list_repo_files=_repo(_NON_GGUF))


def test_an_empty_repository_is_rejected() -> None:
    with pytest.raises(ExperimentError, match="no GGUF files"):
        model_request(REPO, list_repo_files=_repo(()))


@pytest.mark.parametrize(
    ("selector", "expected"),
    [
        ("Q4_K_M", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        ("q4_k_m", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        ("Q8_0", "Llama-3.2-1B-Instruct-Q8_0.gguf"),
        ("Llama-3.2-1B-Instruct-Q4_K_M.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
        ("llama-3.2-1b-instruct-q4_k_m.gguf", "Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    ],
)
def test_a_selector_picks_the_matching_file(selector: str, expected: str) -> None:
    request = model_request(f"{REPO}:{selector}", list_repo_files=_repo([*_QUANTS, *_NON_GGUF]))

    assert request == HubCheckpoint(repo_id=REPO, filename=expected)


def test_a_selector_can_name_a_nested_file_by_its_path() -> None:
    files = ["quants/Llama-3.2-1B-Instruct-Q4_K_M.gguf", "README.md"]

    request = model_request(
        f"{REPO}:quants/Llama-3.2-1B-Instruct-Q4_K_M.gguf", list_repo_files=_repo(files)
    )

    assert request == HubCheckpoint(
        repo_id=REPO, filename="quants/Llama-3.2-1B-Instruct-Q4_K_M.gguf"
    )


def test_a_selector_matching_nothing_is_rejected_with_the_candidates() -> None:
    with pytest.raises(ExperimentError, match="Q5_K_M") as error:
        model_request(f"{REPO}:Q5_K_M", list_repo_files=_repo(_QUANTS))

    assert "Q4_K_M" in str(error.value)
    assert "Q8_0" in str(error.value)


def test_a_selector_matching_several_files_is_rejected_as_ambiguous() -> None:
    files = ("Llama-3.2-1B-Q4_0.gguf", "Llama-3.2-1B-Q4_K_M.gguf")

    with pytest.raises(ExperimentError, match="several GGUF files") as error:
        model_request(f"{REPO}:Q4", list_repo_files=_repo(files))

    assert "Q4_0" in str(error.value)
    assert "Q4_K_M" in str(error.value)


def test_a_selector_is_matched_even_when_the_repository_holds_one_gguf() -> None:
    """An explicit selector that matches nothing is still an error."""
    with pytest.raises(ExperimentError, match="Q4_K_M"):
        model_request(f"{REPO}:Q4_K_M", list_repo_files=_repo(("model-Q8_0.gguf",)))


def test_an_empty_selector_asks_for_the_automatic_pick() -> None:
    request = model_request(f"{REPO}:", list_repo_files=_repo(("model-Q8_0.gguf",)))

    assert request == HubCheckpoint(repo_id=REPO, filename="model-Q8_0.gguf")
