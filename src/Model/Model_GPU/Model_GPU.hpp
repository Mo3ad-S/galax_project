#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_HPP_
#define MODEL_GPU_HPP_

#include "../Model.hpp"

#include <cuda_runtime.h>
#include "kernel.cuh"

class Model_GPU : public Model
{
private:
	// Pinned (page-locked) host memory for fast DMA transfers
	float3* h_positions;
	float3* h_velocities;

	// Device memory
	float3* d_positions;
	float3* d_velocities;
	float3* d_accelerations;
	float*  d_masses;

	// Dedicated CUDA stream for async operations
	cudaStream_t stream;

public:
	Model_GPU(const Initstate& initstate, Particles& particles);

	virtual ~Model_GPU();

	virtual void step();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
