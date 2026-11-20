#!/usr/bin/env bash
set -euo pipefail

# Move to the directory containing this script, wherever it's called from
cd "$(dirname "${BASH_SOURCE[0]}")"

sphinx-build -b html source build