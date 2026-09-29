#!/bin/bash
# Full GaussianShader (BRDF / relightable) run on IMG_6556. See docs/lab-notes.md problem 13.
cd ~/Claude_Test/third_party/GaussianShader
source ~/miniforge3/bin/activate gaussian_shader
export CC=x86_64-conda-linux-gnu-gcc CXX=x86_64-conda-linux-gnu-g++ TORCH_CUDA_ARCH_LIST="8.9" CUDA_HOME=$CONDA_PREFIX
T=$CONDA_PREFIX/targets/x86_64-linux
export CPATH=$T/include LIBRARY_PATH=$T/lib:$T/lib/stubs:/usr/lib/wsl/lib:$CONDA_PREFIX/lib
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== [1/1] GaussianShader IMG_6556, 30000 iters ($(date)) ==="
python train.py -s ~/Claude_Test/data_gaussianshader/IMG_6556 -m ~/Claude_Test/outputs_relightable/IMG_6556 \
  -w --brdf_dim 0 --sh_degree -1 --lambda_predicted_normal 2e-1 --brdf_env 512 \
  -r 2 --data_device cpu --iterations 30000 \
  --test_iterations 7000 15000 30000 --save_iterations 7000 15000 30000 \
  && echo "=== DONE GaussianShader IMG_6556 ($(date)) ===" || echo "=== GaussianShader FAILED ($(date)) ==="
