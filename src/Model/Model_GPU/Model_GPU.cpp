#ifdef GALAX_MODEL_GPU

#include <iostream>

#include "Model_GPU.hpp"
#include "kernel.cuh"

namespace {

inline void check_cuda(cudaError_t status, const char* context)
{
	if (status != cudaSuccess)
		std::cerr << "CUDA error (" << context << "): "
		          << cudaGetErrorString(status) << std::endl;
}

} // namespace

Model_GPU
::Model_GPU(const Initstate& initstate, Particles& particles)
: Model(initstate, particles),
  h_positions(nullptr),  h_velocities(nullptr),
  d_positions(nullptr),  d_velocities(nullptr),
  d_accelerations(nullptr), d_masses(nullptr),
  stream(nullptr)
{
	check_cuda(cudaSetDevice(0), "set device");
	check_cuda(cudaStreamCreate(&stream), "create stream");

	// Pinned host memory enables DMA transfers (roughly 2x bandwidth
	// over pageable memory) and is required for cudaMemcpyAsync.
	check_cuda(cudaMallocHost(&h_positions,  n_particles * sizeof(float3)),
	           "host alloc positions");
	check_cuda(cudaMallocHost(&h_velocities, n_particles * sizeof(float3)),
	           "host alloc velocities");

	std::vector<float> h_masses(n_particles);

	for (int i = 0; i < n_particles; i++)
	{
		h_positions[i]  = make_float3(initstate.positionsx [i],
		                              initstate.positionsy [i],
		                              initstate.positionsz [i]);
		h_velocities[i] = make_float3(initstate.velocitiesx[i],
		                              initstate.velocitiesy[i],
		                              initstate.velocitiesz[i]);
		h_masses[i] = initstate.masses[i];
	}

	// Allocate all device buffers
	check_cuda(cudaMalloc(&d_positions,     n_particles * sizeof(float3)),
	           "alloc d_positions");
	check_cuda(cudaMalloc(&d_velocities,    n_particles * sizeof(float3)),
	           "alloc d_velocities");
	check_cuda(cudaMalloc(&d_accelerations, n_particles * sizeof(float3)),
	           "alloc d_accelerations");
	check_cuda(cudaMalloc(&d_masses,        n_particles * sizeof(float)),
	           "alloc d_masses");

	// Upload initial state to device via async copies on the stream
	check_cuda(cudaMemcpyAsync(d_positions,  h_positions,
	           n_particles * sizeof(float3),
	           cudaMemcpyHostToDevice, stream), "upload positions");

	check_cuda(cudaMemcpyAsync(d_velocities, h_velocities,
	           n_particles * sizeof(float3),
	           cudaMemcpyHostToDevice, stream), "upload velocities");

	check_cuda(cudaMemcpyAsync(d_masses, h_masses.data(),
	           n_particles * sizeof(float),
	           cudaMemcpyHostToDevice, stream), "upload masses");

	check_cuda(cudaMemsetAsync(d_accelerations, 0,
	           n_particles * sizeof(float3), stream), "zero accelerations");

	check_cuda(cudaStreamSynchronize(stream), "init sync");
}

Model_GPU
::~Model_GPU()
{
	if (stream) cudaStreamSynchronize(stream);

	cudaFree(d_positions);
	cudaFree(d_velocities);
	cudaFree(d_accelerations);
	cudaFree(d_masses);

	cudaFreeHost(h_positions);
	cudaFreeHost(h_velocities);

	if (stream) cudaStreamDestroy(stream);
}

void Model_GPU
::step()
{
	// Launch tiled acceleration kernel + velocity/position update on the stream.
	// Both kernels run sequentially on the same stream (implicit ordering,
	// no host-side sync needed between them).
	update_position_cu(d_positions, d_velocities, d_accelerations,
	                   d_masses, n_particles, stream);

	// Async DMA of updated positions back to pinned host memory
	check_cuda(cudaMemcpyAsync(h_positions, d_positions,
	           n_particles * sizeof(float3),
	           cudaMemcpyDeviceToHost, stream), "download positions");

	check_cuda(cudaStreamSynchronize(stream), "step sync");

	// Convert AoS float3 → SoA for the display/validation layer
	for (int i = 0; i < n_particles; i++)
	{
		particles.x[i] = h_positions[i].x;
		particles.y[i] = h_positions[i].y;
		particles.z[i] = h_positions[i].z;
	}
}

#endif // GALAX_MODEL_GPU
