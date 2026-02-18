#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>
#include <algorithm>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;

// Use the best available SIMD architecture
using batch_type = xs::batch<float>;
constexpr size_t simd_size = batch_type::size;

Model_CPU_fast
::Model_CPU_fast(const Initstate& initstate, Particles& particles)
: Model_CPU(initstate, particles)
{
}

void Model_CPU_fast
::step()
{
    // Zero out accelerations
    std::fill(accelerationsx.begin(), accelerationsx.end(), 0.0f);
    std::fill(accelerationsy.begin(), accelerationsy.end(), 0.0f);
    std::fill(accelerationsz.begin(), accelerationsz.end(), 0.0f);

    // Constants for the force calculation
    const batch_type one(1.0f);
    const batch_type ten(10.0f);

    // Compute gravitational forces using OpenMP + SIMD
    // Strategy: Parallelize outer loop (particle i), vectorize inner loop (particle j)
    #pragma omp parallel for schedule(dynamic, 64)
    for (int i = 0; i < n_particles; i++)
    {
        // Load position of particle i (scalar - broadcast to SIMD)
        const batch_type pos_i_x(particles.x[i]);
        const batch_type pos_i_y(particles.y[i]);
        const batch_type pos_i_z(particles.z[i]);
        
        // Accumulators for particle i's acceleration
        batch_type acc_i_x(0.0f);
        batch_type acc_i_y(0.0f);
        batch_type acc_i_z(0.0f);

        // Vectorized inner loop - process simd_size particles at a time
        int j = 0;
        const int n_simd_chunks = (n_particles / simd_size) * simd_size;
        
        for (j = 0; j < n_simd_chunks; j += simd_size)
        {
            // Load positions of simd_size particles j
            const batch_type pos_j_x = batch_type::load_unaligned(&particles.x[j]);
            const batch_type pos_j_y = batch_type::load_unaligned(&particles.y[j]);
            const batch_type pos_j_z = batch_type::load_unaligned(&particles.z[j]);
            
            // Load masses of particles j
            const batch_type mass_j = batch_type::load_unaligned(&initstate.masses[j]);
            
            // Compute distance vector: r_ij = r_j - r_i
            const batch_type diff_x = pos_j_x - pos_i_x;
            const batch_type diff_y = pos_j_y - pos_i_y;
            const batch_type diff_z = pos_j_z - pos_i_z;
            
            // Compute squared distance: |r_ij|^2
            batch_type dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
            
            // Apply softening and compute force factor
            // if dist_sq < 1, use factor = 10.0
            // else factor = 10.0 / (dist^3) = 10.0 / (dist_sq^1.5)
            auto mask_close = dist_sq < one;
            
            // For distant particles: compute 1/r^3
            batch_type dist = xs::sqrt(dist_sq);
            batch_type inv_dist_cubed = ten / (dist_sq * dist);
            
            // Apply softening: if too close, use constant factor
            batch_type force_factor = xs::select(mask_close, ten, inv_dist_cubed);
            
            // Create mask to exclude self-interaction (i == j)
            // Check if any j in this batch equals i
            alignas(32) float force_array[simd_size];
            force_factor.store_aligned(force_array);
            
            for (size_t k = 0; k < simd_size; ++k)
            {
                if (j + k == i)
                {
                    force_array[k] = 0.0f;
                }
            }
            
            force_factor = batch_type::load_aligned(force_array);
            
            // Multiply by mass: F = m_j * factor * r_ij
            const batch_type weighted_factor = force_factor * mass_j;
            
            // Accumulate accelerations
            acc_i_x += diff_x * weighted_factor;
            acc_i_y += diff_y * weighted_factor;
            acc_i_z += diff_z * weighted_factor;
        }
        
        // Handle remaining particles (scalar tail)
        for (; j < n_particles; j++)
        {
            if (i != j)
            {
                const float diff_x = particles.x[j] - particles.x[i];
                const float diff_y = particles.y[j] - particles.y[i];
                const float diff_z = particles.z[j] - particles.z[i];
                
                float dist_sq = diff_x * diff_x + diff_y * diff_y + diff_z * diff_z;
                
                float force_factor;
                if (dist_sq < 1.0f)
                {
                    force_factor = 10.0f;
                }
                else
                {
                    const float dist = std::sqrt(dist_sq);
                    force_factor = 10.0f / (dist_sq * dist);
                }
                
                const float weighted_factor = force_factor * initstate.masses[j];
                accelerationsx[i] += diff_x * weighted_factor;
                accelerationsy[i] += diff_y * weighted_factor;
                accelerationsz[i] += diff_z * weighted_factor;
            }
        }
        
        // Reduce SIMD accumulators to scalar and store
        accelerationsx[i] += xs::hadd(acc_i_x);
        accelerationsy[i] += xs::hadd(acc_i_y);
        accelerationsz[i] += xs::hadd(acc_i_z);
    }

    // Update velocities and positions
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_particles; i++)
    {
        velocitiesx[i] += accelerationsx[i] * 2.0f;
        velocitiesy[i] += accelerationsy[i] * 2.0f;
        velocitiesz[i] += accelerationsz[i] * 2.0f;
        
        particles.x[i] += velocitiesx[i] * 0.1f;
        particles.y[i] += velocitiesy[i] * 0.1f;
        particles.z[i] += velocitiesz[i] * 0.1f;
    }
}

#endif // GALAX_MODEL_CPU_FAST
