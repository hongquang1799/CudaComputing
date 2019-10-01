#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "iostream"
#include "time.h"

#include <chrono>
#include <stdio.h>

#include <assert.h>

float cpu_sigmoid(float x)
{
	return 1.0f / (1.0f + expf(x));
}

void CPU_AddVector(float * a, float * b, float * c, float n)
{
	int i;

	for (i = 0; i < n; i++)
	{
		c[i] = (a[i] + b[i]);
	}
}

__device__ float gpu_sigmoid(float x)
{
	return 1.0f / (1.0f + expf(x));
}

__global__ void GPU_AddVector(float * a, float * b, float * c, int n)
{
	int i = blockIdx.x*blockDim.x + threadIdx.x;

	if (i < n)
		c[i] = (a[i] + b[i]);
}

void main()
{
	srand(time(NULL));

	size_t n = 1 << 10;
	size_t n_bytes = sizeof(float) * n;

	float * a = (float *)malloc(n_bytes);
	float * b = (float *)malloc(n_bytes);
	float * c = (float *)malloc(n_bytes);

	float *d_a, *d_b, *d_c;
	
	cudaMalloc(&d_a, n_bytes);
	cudaMalloc(&d_b, n_bytes);
	cudaMalloc(&d_c, n_bytes);

	for (int i = 0; i < n; i++)
	{
		a[i] = rand() % 10;
		b[i] = rand() % 10;
	}

	cudaMemcpy(d_a, a, n_bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, n_bytes, cudaMemcpyHostToDevice);

	size_t n_iter = 10000;

	// cpu computing
	auto start = std::chrono::high_resolution_clock::now();
	
	for (int i = 0; i < n_iter; i++)
	{
		CPU_AddVector(a, b, c, n);
	}
		
	
	auto stop = std::chrono::high_resolution_clock::now();
	auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Speed of CPU: " << duration.count() << " micro seconds" << std::endl;


	// gpu computing
	GPU_AddVector << <1, 1024>> >(d_a, d_b, d_c, n);
	start = std::chrono::high_resolution_clock::now();

	for (int i = 0; i < n_iter; i++)
	{
		GPU_AddVector<<<1, 1024>>>(d_a, d_b, d_c, n);
	}
	cudaDeviceSynchronize();

	stop = std::chrono::high_resolution_clock::now();
	duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
	std::cout << "Speed of GPU: " << duration.count() << " micro seconds" << std::endl;


	// Warmup
	//cudaEvent_t launch_begin, launch_end;
	//cudaEventCreate(&launch_begin);
	//cudaEventCreate(&launch_end);
	//GPU_AddVector<<<1, n>>>(d_a, d_b, d_c, n);// num blocks, num_threads
	//float total_time = 0;
	//int num_times = 100;
	//// Get average of 100 runs
	//for (int i = 0; i<num_times; i++) {
	//	cudaEventRecord(launch_begin, 0);
	//	GPU_AddVector <<<1, n >>>(d_a, d_b, d_c, n);
	//	cudaEventRecord(launch_end, 0);
	//	cudaEventSynchronize(launch_end);
	//	float time = 0;
	//	cudaEventElapsedTime(&time, launch_begin, launch_end);
	//	total_time += time;
	//}
	//std::cout << "Speed of CPU vector Addition: " << total_time << " micro seconds" << std::endl;

	cudaMemcpy(c, d_c, n_bytes, cudaMemcpyDeviceToHost);

	for (int i = 0; i < 5; i++)
	{
		printf("%f + %f = %f\n", a[i], b[i], c[i]);
	}



	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	free(a);
	free(b);
	free(c);

	getchar();
}