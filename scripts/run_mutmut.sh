#!/bin/bash
# Run mutation testing on specific modules.
# Usage: ./scripts/run_mutmut.sh [module_path]
# Example: ./scripts/run_mutmut.sh fastcode/src/fastcode/scip_models.py
set -e

MODULE="${1:-fastcode/src/fastcode/scip_models.py}"
echo "Running mutmut on $MODULE"

mutmut run \
  --paths-to-mutate "$MODULE" \
  --runner "uv run pytest fastcode/tests/ -x -q --tb=short -o 'addopts=' -k 'not slow'" \
  --use-coverage
