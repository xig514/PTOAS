"""Section context managers: section_vector and section_cube.

These emit ``pto.section.vector`` / ``pto.section.cube`` ops whose bodies
are lowered to ``#if defined(__DAV_VEC__)`` / ``#if defined(__DAV_CUBE__)``
guards in the generated C++ code.

Usage::

    with pto.section_vector():
        # ops inside are guarded by #if defined(__DAV_VEC__)
        tile = pto.make_tile(...)
        pto.tload(tile, pv)
        ...

    with pto.section_cube():
        # ops inside are guarded by #if defined(__DAV_CUBE__)
        ...
"""

from contextlib import contextmanager

from mlir.ir import InsertionPoint
from mlir.dialects import pto as _pto


@contextmanager
def section_vector():
    """Emit a ``pto.section.vector`` region.

    All ops created inside the ``with`` block are placed in the
    section body, which lowers to ``#if defined(__DAV_VEC__)`` in C++.
    """
    op = _pto.SectionVectorOp()
    block = op.body.blocks.append()
    ip = InsertionPoint(block)
    ip.__enter__()
    try:
        yield
    finally:
        ip.__exit__(None, None, None)


@contextmanager
def section_cube():
    """Emit a ``pto.section.cube`` region.

    All ops created inside the ``with`` block are placed in the
    section body, which lowers to ``#if defined(__DAV_CUBE__)`` in C++.
    """
    op = _pto.SectionCubeOp()
    block = op.body.blocks.append()
    ip = InsertionPoint(block)
    ip.__enter__()
    try:
        yield
    finally:
        ip.__exit__(None, None, None)
