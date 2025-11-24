#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)

template <typename T> 
void check(T err, const char* const func, const char* const file, const int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(err), cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}

int main(void){
    int numElements = 5000; 
    size_t size = numElements * sizeof(float); 
    float *h_A, *h_B, *h_C; 
    float *d_A, *d_B, *d_C; 
    cudaStream_t stream1, stream2; 
    
    h_A = (float *)malloc(size);
    h_B = (float *)malloc(size); 
    h_C = (float *)malloc(size); 
    
    for (int i = 0; i < numElements; ++i){
        h_A[i] = rand() / (float)RAND_MAX;
        h_B[i] = rand() / (float)RAND_MAX;
    }
    
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_A, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_B, size));
    CHECK_CUDA_ERROR(cudaMalloc((void **)&d_C, size));
    
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream1));
    CHECK_CUDA_ERROR(cudaStreamCreate(&stream2));
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_A, h_A, size, cudaMemcpyHostToDevice, stream1));
    CHECK_CUDA_ERROR(cudaMemcpyAsync(d_B, h_B, size, cudaMemcpyHostToDevice, stream2));
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C, numElements);
    
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h_C, d_C, size, cudaMemcpyDeviceToHost, stream1));
    
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream1));
    CHECK_CUDA_ERROR(cudaStreamSynchronize(stream2));
    
    CHECK_CUDA_ERROR(cudaFree(d_A));
    CHECK_CUDA_ERROR(cudaFree(d_B));
    CHECK_CUDA_ERROR(cudaFree(d_C));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream1));
    CHECK_CUDA_ERROR(cudaStreamDestroy(stream2));
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}