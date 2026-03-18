#ifdef GALAX_MODEL_GPU

#ifndef __KERNEL_CUH__
#define __KERNEL_CUH__

#include <stdio.h>
#include <cuda_runtime.h>

void update_position_cu(float3* positionsGPU, float3* velocitiesGPU, float3* accelerationsGPU, float* massesGPU, int n_particles, cudaStream_t stream = 0);

#endif

#endif // GALAX_MODEL_GPU
