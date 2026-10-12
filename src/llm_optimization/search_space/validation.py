"""Engine-support validation for an experiment search-space definition.

Parsing a definition checks it engine-agnostically: the shape of the file and
the kind of every declared value. It cannot know whether a particular engine
forwards a parameter or supports a value, because that depends on the backend
the caller will run with. That second question is asked explicitly, after the
space has been parsed, through :func:`validate_space`.

The function is a thin aggregator over the backend's own
:meth:`~llm_optimization.inference.types.InferenceBackend.validate` hook: the
backend reports every problem it sees, and they are collected into a single
:class:`~llm_optimization.search_space.types.ExperimentDefinitionError`, so an
unsupported space fails once, up front, exactly like a malformed one.
"""

from typing import TYPE_CHECKING

from .types import ExperimentDefinitionError, ExperimentSpace

if TYPE_CHECKING:
    from ..inference.types import InferenceBackend


def validate_space(space: ExperimentSpace, backend: InferenceBackend) -> None:
    """Check that ``backend`` supports every dimension of ``space``.

    The space is assumed to have passed engine-agnostic validation when it was
    parsed. The backend's
    :meth:`~llm_optimization.inference.types.InferenceBackend.validate` hook is
    asked for every problem it sees, and they are collected into one
    :class:`~llm_optimization.search_space.types.ExperimentDefinitionError` so
    the caller can correct the whole definition in a single pass. An empty
    result returns ``None``, meaning the backend supports the space.
    """
    problems = tuple(backend.validate(space))
    if problems:
        raise ExperimentDefinitionError(problems)
