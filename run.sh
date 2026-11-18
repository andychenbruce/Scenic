#!/usr/bin/env bash
set -e

cd "$(dirname "$0")"

# Scenic renders scenes with PyTorch3D's differentiable rasterizer when it is
# available.  Prefer the project virtualenv, then a dedicated PyTorch3D
# environment; if neither has PyTorch3D, fall back to trimesh's OpenGL viewer.
PYTHON="${SCENIC_PYTHON:-my_venv/bin/python}"
if ! "$PYTHON" -c "import pytorch3d" >/dev/null 2>&1; then
    if [ -x /opt/p3d-env/bin/python ] && /opt/p3d-env/bin/python -c "import pytorch3d" >/dev/null 2>&1; then
        PYTHON=/opt/p3d-env/bin/python
    fi
fi

rm -f output_*.png

if "$PYTHON" -c "import pytorch3d" >/dev/null 2>&1; then
    "$PYTHON" -m scenic --count 10 ./test.sc
else
    echo "warning: PyTorch3D not found; falling back to the trimesh viewer" >&2
    xvfb-run "$PYTHON" -m scenic --count 10 ./test.sc
fi
