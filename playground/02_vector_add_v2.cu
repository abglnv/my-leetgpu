#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 100000
#define BLOCK_SIZE_1D 1024

#define NX 100
#define NY 100
#define NZ 10
#define BLOCK_SIZE_3D_X 16
#define BLOCK_SIZE_3D_Y 8
#define BLOCK_SIZE_3D_Z 8
// Total threads per 3D block: 16 * 8 * 8 = 1024 (Good size)

void vector_add_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_1d(float *a, float *b, float *c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

__global__ void vector_add_gpu_3d(float *a, float *b, float *c, int nx, int ny, int nz) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < nx && j < ny && k < nz) {
        int idx = i + j * nx + k * nx * ny;

        c[idx] = a[idx] + b[idx];
    }
}

void init_vector(float *vec, int n) {
    for (int i = 0; i < n; i++) {
        vec[i] = (float)rand() / RAND_MAX;
    }
}

double get_time(){
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    float *h_a, *h_b, *h_c_cpu, *h_c_gpu_1d, *h_c_gpu_3d;
    float *d_a, *d_b, *d_c_1d, *d_c_3d;
    size_t size = N * sizeof(float);

    h_a = (float*)malloc(size);
    h_b = (float*)malloc(size);
    h_c_cpu = (float*)malloc(size);
    h_c_gpu_1d = (float*)malloc(size);
    h_c_gpu_3d = (float*)malloc(size);

    srand(time(NULL));
    init_vector(h_a, N);
    init_vector(h_b, N);

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c_1d, size);
    cudaMalloc(&d_c_3d, size);

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int num_blocks_1d = (N + BLOCK_SIZE_1D - 1) / BLOCK_SIZE_1D;

    int nx = NX, ny = NY, nz = NZ;

    dim3 block_size_3d(BLOCK_SIZE_3D_X, BLOCK_SIZE_3D_Y, BLOCK_SIZE_3D_Z);
    dim3 num_blocks_3d(
        (nx + block_size_3d.x - 1) / block_size_3d.x,
        (ny + block_size_3d.y - 1) / block_size_3d.y,
        (nz + block_size_3d.z - 1) / block_size_3d.z
    );

    printf("--- Benchmark Configuration ---\n");
    printf("Total Elements (N): %d\n", N);
    printf("1D Grid: %d blocks of %d threads\n", num_blocks_1d, BLOCK_SIZE_1D);
    printf("3D Grid: %dx%dx%d blocks of %dx%dx%d threads\n",
           (int)num_blocks_3d.x, (int)num_blocks_3d.y, (int)num_blocks_3d.z,
           (int)block_size_3d.x, (int)block_size_3d.y, (int)block_size_3d.z);
    printf("Total threads launched (1D/3D): %d\n\n",
           (int)num_blocks_1d * BLOCK_SIZE_1D);


    printf("Performing warm-up runs (3 iterations)...\n");
    for (int i = 0; i < 3; i++) {
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
    }
    printf("Warm-up complete.\n\n");

    // CPU
    printf("Benchmarking CPU implementation (5 runs)...\n");
    double cpu_total_time = 0.0;
    for (int i = 0; i < 5; i++) {
        double start_time = get_time();
        vector_add_cpu(h_a, h_b, h_c_cpu, N);
        double end_time = get_time();
        cpu_total_time += end_time - start_time;
    }
    double cpu_avg_time = cpu_total_time / 5.0;

    // GPU 1D
    printf("Benchmarking GPU 1D implementation (10 runs)...\n");
    double gpu_1d_total_time = 0.0;
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_c_1d, 0, size);
        double start_time = get_time();
        vector_add_gpu_1d<<<num_blocks_1d, BLOCK_SIZE_1D>>>(d_a, d_b, d_c_1d, N);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_1d_total_time += end_time - start_time;
    }
    double gpu_1d_avg_time = gpu_1d_total_time / 10.0;

    // GPU 3D
    printf("Benchmarking GPU 3D implementation (10 runs)...\n");
    double gpu_3d_total_time = 0.0;
    for (int i = 0; i < 10; i++) {
        cudaMemset(d_c_3d, 0, size);
        double start_time = get_time();
        vector_add_gpu_3d<<<num_blocks_3d, block_size_3d>>>(d_a, d_b, d_c_3d, nx, ny, nz);
        cudaDeviceSynchronize();
        double end_time = get_time();
        gpu_3d_total_time += end_time - start_time;
    }
    double gpu_3d_avg_time = gpu_3d_total_time / 10.0;

    printf("\n--- Results ---\n");
    printf("CPU average time:      %f milliseconds\n", cpu_avg_time * 1000);
    printf("GPU 1D average time:   %f milliseconds\n", gpu_1d_avg_time * 1000);
    printf("GPU 3D average time:   %f milliseconds\n", gpu_3d_avg_time * 1000);
    printf("Speedup (CPU vs GPU 1D): %8.2fx\n", cpu_avg_time / gpu_1d_avg_time);
    printf("Speedup (CPU vs GPU 3D): %8.2fx\n", cpu_avg_time / gpu_3d_avg_time);
    printf("Speedup (GPU 1D vs GPU 3D): %8.2fx\n", gpu_1d_avg_time / gpu_3d_avg_time);

    free(h_a);
    free(h_b);
    free(h_c_cpu);
    free(h_c_gpu_1d);
    free(h_c_gpu_3d);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c_1d);
    cudaFree(d_c_3d);

    return 0;
}
