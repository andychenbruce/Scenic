#!/usr/bin/env bash
# Create a Python 3.12 environment with a CUDA-enabled PyTorch3D build.
#
# PyTorch3D publishes Linux builds for Python <= 3.12 (and a limited set of
# PyTorch versions), so it cannot be installed into Scenic's default Python
# 3.14 virtualenv.  This script uses micromamba to create a separate
# environment; `run.sh` discovers it automatically.
#
# Usage: tools/setup_pytorch3d_env.sh [ENV_DIR]   (default: /opt/p3d-env)
set -euo pipefail

REPO_ROOT=$(cd "$(dirname "$0")/.." && pwd)
ENV_DIR=${1:-/opt/p3d-env}
export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/opt/mamba}

if ! command -v micromamba >/dev/null 2>&1; then
    echo "Installing micromamba to /usr/local/bin..."
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest |
        tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba
    chmod +x /usr/local/bin/micromamba
fi

micromamba create -y -p "$ENV_DIR" \
    -c pytorch3d -c pytorch -c nvidia -c conda-forge \
    python=3.12 pytorch=2.4.1 pytorch-cuda=12.1 pytorch3d=0.7.8

# Install Scenic and its dependencies; the conda-provided PyTorch stays put.
"$ENV_DIR/bin/pip" install -e "$REPO_ROOT"
"$ENV_DIR/bin/pip" install "opencv-python~=4.5" "pyglet>=1.5,<=1.5.26"

"$ENV_DIR/bin/python" - <<'EOF'
import torch
import pytorch3d
import scenic

print(f"torch {torch.__version__}, pytorch3d {pytorch3d.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF
