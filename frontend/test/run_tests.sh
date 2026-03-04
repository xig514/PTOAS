#!/usr/bin/env bash
# Run all pto_frontend tests.
#
# Each test generates MLIR via the frontend then compiles with ptoas.
# Usage:  cd frontend/test && bash run_tests.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PASS=0
FAIL=0

run_test() {
    local name="$1"
    local script="$SCRIPT_DIR/$name"
    local pto_file="/tmp/pto_frontend_${name%.py}.pto"
    echo -n "  $name ... "
    if python3 "$script" > "$pto_file" 2>&1; then
        if ptoas "$pto_file" --pto-level=level3 > /dev/null 2>&1; then
            echo "PASS"
            PASS=$((PASS + 1))
        else
            echo "FAIL (ptoas compilation)"
            FAIL=$((FAIL + 1))
        fi
    else
        echo "FAIL (IR generation)"
        cat "$pto_file" || true
        FAIL=$((FAIL + 1))
    fi
}

echo "=== pto_frontend test suite ==="
run_test test_vector_add.py
run_test test_matmul.py
run_test test_dynamic_shape.py
run_test test_control_flow.py
echo ""
echo "Results: $PASS passed, $FAIL failed"
[ "$FAIL" -eq 0 ]
