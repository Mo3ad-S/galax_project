#ifdef GALAX_MODEL_GPU

#include "kernel.cuh"
#include <cuda_runtime.h>

#define TILE_SIZE 256

// N-body acceleration kernel using shared memory tiling.
// Each block cooperatively loads tiles of (position + mass) into shared memory
// as packed float4 values, reducing global memory bandwidth by a factor of
// TILE_SIZE compared to the naive approach.
__global__ void __launch_bounds__(TILE_SIZE)
compute_acc_tiled(const float3* __restrict__ positions,
                  float3*       __restrict__ accelerations,
                  const float*  __restrict__ masses,
                  int n_particles)
{
    __shared__ float4 tile[TILE_SIZE];

    const unsigned int i   = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int tid = threadIdx.x;

    const float3 my_pos = (i < n_particles) ? positions[i]
                                            : make_float3(0.0f, 0.0f, 0.0f);
    float ax = 0.0f, ay = 0.0f, az = 0.0f;

    for (int tile_start = 0; tile_start < n_particles; tile_start += TILE_SIZE)
    {
        const int load_idx = tile_start + tid;
        if (load_idx < n_particles)
        {
            const float3 p = positions[load_idx];
            tile[tid] = make_float4(p.x, p.y, p.z, masses[load_idx]);
        }
        else
        {
            tile[tid] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
        __syncthreads();

        if (i < n_particles)
        {
            const int tile_end = min(TILE_SIZE, n_particles - tile_start);

            #pragma unroll 16
            for (int j = 0; j < tile_end; j++)
            {
                const float4 tj = tile[j];
                const float dx = tj.x - my_pos.x;
                const float dy = tj.y - my_pos.y;
                const float dz = tj.z - my_pos.z;

                float dist_sq = fmaf(dx, dx, fmaf(dy, dy, dz * dz));

                // Softening: constant force below threshold, inverse-cube above.
                // Self-interaction (i==j) gives diff=(0,0,0), contributing zero
                // regardless of the force factor — no branch needed.
                float factor;
                if (dist_sq < 1.0f)
                    factor = 10.0f;
                else
                {
                    float inv_dist = rsqrtf(dist_sq);
                    factor = 10.0f * inv_dist * inv_dist * inv_dist;
                }

                const float w = factor * tj.w;
                ax = fmaf(dx, w, ax);
                ay = fmaf(dy, w, ay);
                az = fmaf(dz, w, az);
            }
        }
        __syncthreads();
    }

    if (i < n_particles)
        accelerations[i] = make_float3(ax, ay, az);
}

__global__ void __launch_bounds__(TILE_SIZE)
update_vel_pos(float3* __restrict__ positions,
               float3* __restrict__ velocities,
               const float3* __restrict__ accelerations,
               int n_particles)
{
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_particles)
        return;

    float3 vel = velocities[i];
    const float3 acc = accelerations[i];

    vel.x = fmaf(acc.x, 2.0f, vel.x);
    vel.y = fmaf(acc.y, 2.0f, vel.y);
    vel.z = fmaf(acc.z, 2.0f, vel.z);

    float3 pos = positions[i];
    pos.x = fmaf(vel.x, 0.1f, pos.x);
    pos.y = fmaf(vel.y, 0.1f, pos.y);
    pos.z = fmaf(vel.z, 0.1f, pos.z);

    velocities[i] = vel;
    positions[i]  = pos;
}

void update_position_cu(float3* positionsGPU,
                        float3* velocitiesGPU,
                        float3* accelerationsGPU,
                        float*  massesGPU,
                        int     n_particles,
                        cudaStream_t stream)
{
    const int nthreads = TILE_SIZE;
    const int nblocks  = (n_particles + nthreads - 1) / nthreads;

    compute_acc_tiled<<<nblocks, nthreads, 0, stream>>>(
        positionsGPU, accelerationsGPU, massesGPU, n_particles);

    update_vel_pos<<<nblocks, nthreads, 0, stream>>>(
        positionsGPU, velocitiesGPU, accelerationsGPU, n_particles);
}

#endif // GALAX_MODEL_GPU
