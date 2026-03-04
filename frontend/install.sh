#!/usr/bin/env bash
# Install pto_frontend in editable (development) mode.
#
# Prerequisites:
#   - PYTHONPATH must include the MLIR and PTO Python packages
#     (see CLAUDE.md "Full configure" for the exact exports).
#   - LD_LIBRARY_PATH must include the LLVM and PTO shared libs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
pip install -e "$SCRIPT_DIR"

echo ""
echo "pto_frontend installed.  Make sure PYTHONPATH includes:"
echo "  \$MLIR_PYTHON_ROOT  (mlir core python packages)"
echo "  \$PTO_PYTHON_ROOT   (PTO python bindings)"
