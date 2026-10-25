"""Unit tests for the writer registry and its selection entry point.

The registry is the seam between an :class:`OutputFormat` chosen on the command
line and the renderer that produces the artifact. These tests pin that every
format has exactly one registered writer and that each writer declares whether
the composition root must plan the server before rendering.
"""

from llm_config_optimization.output import (
    WRITERS,
    ComposeWriter,
    ConfigWriter,
    OutputFormat,
    QuadletWriter,
    YamlWriter,
    writer_for,
)


def test_every_output_format_is_registered() -> None:
    """The registry is exhaustive over the selectable formats."""
    assert set(WRITERS) == set(OutputFormat)


def test_writer_for_selects_the_registered_writer() -> None:
    """Selection is a lookup on the format enum and returns its writer."""
    for format in OutputFormat:
        writer = writer_for(format)
        assert writer is WRITERS[format]
        assert isinstance(writer, ConfigWriter)


def test_each_format_selects_its_own_renderer() -> None:
    """The three formats map to the three distinct writer implementations."""
    assert isinstance(writer_for(OutputFormat.YAML), YamlWriter)
    assert isinstance(writer_for(OutputFormat.QUADLET), QuadletWriter)
    assert isinstance(writer_for(OutputFormat.COMPOSE), ComposeWriter)


def test_only_the_yaml_writer_renders_without_a_plan() -> None:
    """Executable artifacts need the planned server; the description does not."""
    assert {format: writer.requires_plan for format, writer in WRITERS.items()} == {
        OutputFormat.YAML: False,
        OutputFormat.QUADLET: True,
        OutputFormat.COMPOSE: True,
    }
