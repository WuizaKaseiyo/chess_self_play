#!/bin/bash
# Wrapper run by condor (executable=/bin/bash, arguments=this script).
# Activates the chess venv and runs the env smoke test on the GPU node.
set -x
source $HOME/envs/chess/bin/activate
cd $HOME/verl-agent
export PYTHONPATH=$HOME/verl-agent:${PYTHONPATH}
export TMPDIR=$HOME/fast/tmp

echo "host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

python recipe/hgpo/test_env.py
