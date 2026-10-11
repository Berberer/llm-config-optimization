"""Pytest configuration for the test suite.

The presence of this file puts the ``tests`` directory on ``sys.path`` (pytest
prepends the rootdir of each conftest), so shared test doubles in
``tests/fakes.py`` are importable as ``fakes`` from every test package without
turning the test tree into an installed package.
"""
