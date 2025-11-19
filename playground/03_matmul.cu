#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>

#define M 256 // number of rows in A&C
#define K 512 // number of columns in A and rows in B
#define N 256 // number of columns in B&C
#define BLOCK_SIZE 32 

// CPU matrix multiplication 
void matmul_cpu(float *A, float *B, float *C, int m, int k, int n){
    for (int i = 0; i < m; i++){
        for (int j = 0; j < n; j++){
            float sum = 0.0f; 
            for (int l = 0; l < k; l++){
                sum += A[i*k + l] * B[l*n+j]; 
            }
            C[i*n+j] = sum; 
        }
    }
}

// GPU matrix multiplication 
__global__ void matmul_gpu(float *A, float *B, float *C, int m, int k, int n){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n){
        float sum = 0.0f; 
        for (int l = 0; l < k; l++){
            sum += A[row * k + l] * B[l * n + col]; 
        }
        C[row * n + col] = sum; 
    }
}