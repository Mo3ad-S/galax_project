#ifdef GALAX_MODEL_CPU_FAST

#include <cmath>

#include "Model_CPU_fast.hpp"

#include <xsimd/xsimd.hpp>
#include <omp.h>

namespace xs = xsimd;

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
    const float* __restrict__ px = particles.x.data();
    const float* __restrict__ py = particles.y.data();
    const float* __restrict__ pz = particles.z.data();
    const float* __restrict__ ms = initstate.masses.data();

    const batch_type v_one(1.0f);
    const batch_type v_ten(10.0f);

    const int n_simd = static_cast<int>((n_particles / simd_size) * simd_size);

    // Force computation — O(N^2) pairwise interactions
    // Each outer iteration is independent: static scheduling, no fill needed
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_particles; i++)
    {
        const batch_type pos_i_x(px[i]);
        const batch_type pos_i_y(py[i]);
        const batch_type pos_i_z(pz[i]);

        batch_type acc_x(0.0f);
        batch_type acc_y(0.0f);
        batch_type acc_z(0.0f);

        for (int j = 0; j < n_simd; j += simd_size)
        {
            const batch_type pos_j_x = batch_type::load_unaligned(&px[j]);
            const batch_type pos_j_y = batch_type::load_unaligned(&py[j]);
            const batch_type pos_j_z = batch_type::load_unaligned(&pz[j]);
            const batch_type mass_j  = batch_type::load_unaligned(&ms[j]);

            const batch_type diff_x = pos_j_x - pos_i_x;
            const batch_type diff_y = pos_j_y - pos_i_y;
            const batch_type diff_z = pos_j_z - pos_i_z;

            // dist_sq = dx^2 + dy^2 + dz^2 using fma
            const batch_type dist_sq = xs::fma(diff_x, diff_x,
                                       xs::fma(diff_y, diff_y,
                                               diff_z * diff_z));

            // Softening: if dist_sq < 1 use constant factor, else 10/dist^3
            const auto mask_close = dist_sq < v_one;

            // rsqrt is ~5 cycles vs sqrt(~15) + div(~15) = ~30 cycles
            // 10/dist^3 = 10 * rsqrt(dist_sq)^3
            const batch_type rsqrt_val = xs::rsqrt(dist_sq);
            const batch_type inv_dist_cubed = v_ten * rsqrt_val * rsqrt_val * rsqrt_val;

            const batch_type force_factor = xs::select(mask_close, v_ten, inv_dist_cubed);

            // Self-interaction (i==j) contributes zero: diff is (0,0,0),
            // so diff * anything = 0. No masking needed.
            const batch_type weighted = force_factor * mass_j;

            acc_x = xs::fma(diff_x, weighted, acc_x);
            acc_y = xs::fma(diff_y, weighted, acc_y);
            acc_z = xs::fma(diff_z, weighted, acc_z);
        }

        float ax = xs::hadd(acc_x);
        float ay = xs::hadd(acc_y);
        float az = xs::hadd(acc_z);

        for (int j = n_simd; j < n_particles; j++)
        {
            if (i != j)
            {
                const float dx = px[j] - px[i];
                const float dy = py[j] - py[i];
                const float dz = pz[j] - pz[i];

                float d2 = dx * dx + dy * dy + dz * dz;
                float ff;
                if (d2 < 1.0f)
                    ff = 10.0f;
                else
                {
                    const float d = std::sqrt(d2);
                    ff = 10.0f / (d2 * d);
                }

                const float w = ff * ms[j];
                ax += dx * w;
                ay += dy * w;
                az += dz * w;
            }
        }

        accelerationsx[i] = ax;
        accelerationsy[i] = ay;
        accelerationsz[i] = az;
    }

    // Velocity and position integration — vectorized with SIMD + fma
    const batch_type dt_acc(2.0f);
    const batch_type dt_pos(0.1f);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n_simd; i += simd_size)
    {
        batch_type vx = batch_type::load_unaligned(&velocitiesx[i]);
        batch_type vy = batch_type::load_unaligned(&velocitiesy[i]);
        batch_type vz = batch_type::load_unaligned(&velocitiesz[i]);

        vx = xs::fma(batch_type::load_unaligned(&accelerationsx[i]), dt_acc, vx);
        vy = xs::fma(batch_type::load_unaligned(&accelerationsy[i]), dt_acc, vy);
        vz = xs::fma(batch_type::load_unaligned(&accelerationsz[i]), dt_acc, vz);

        vx.store_unaligned(&velocitiesx[i]);
        vy.store_unaligned(&velocitiesy[i]);
        vz.store_unaligned(&velocitiesz[i]);

        xs::fma(vx, dt_pos, batch_type::load_unaligned(&particles.x[i])).store_unaligned(&particles.x[i]);
        xs::fma(vy, dt_pos, batch_type::load_unaligned(&particles.y[i])).store_unaligned(&particles.y[i]);
        xs::fma(vz, dt_pos, batch_type::load_unaligned(&particles.z[i])).store_unaligned(&particles.z[i]);
    }

    for (int i = n_simd; i < n_particles; i++)
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
