#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <cmath>

#define ROWS 1024
#define COLS 1024
#define ITERS 100

#define BX 32
#define BY 32
#define TIME_TILE 4

#define IDX(i, j) ((i) * COLS + (j))

/************************ CPU REFERENCE *************************/
void cpu_stencil(float *A, float *B)
{
    for (int t = 0; t < ITERS; t++)
    {
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                int top = max(i - 1, 0);
                int bottom = min(i + 1, ROWS - 1);
                int left = max(j - 1, 0);
                int right = min(j + 1, COLS - 1);

                A[IDX(i, j)] =
                    B[IDX(i, j)] +
                    B[IDX(top, j)] +
                    B[IDX(bottom, j)] +
                    B[IDX(i, left)] +
                    B[IDX(i, right)];
            }
        }
        std::swap(A, B);
    }
}

/**************** CASE 1: GLOBAL, NO TRAPEZOID ******************/
__global__ void stencil_global(float *A, float *B)
{
    int i = blockIdx.y * BY + threadIdx.y;
    int j = blockIdx.x * BX + threadIdx.x;
    if (i >= ROWS || j >= COLS)
        return;

    int top = max(i - 1, 0);
    int bottom = min(i + 1, ROWS - 1);
    int left = max(j - 1, 0);
    int right = min(j + 1, COLS - 1);

    A[IDX(i, j)] =
        B[IDX(i, j)] +
        B[IDX(top, j)] +
        B[IDX(bottom, j)] +
        B[IDX(i, left)] +
        B[IDX(i, right)];
}

/**************** CASE 2: SHARED, NO TRAPEZOID ******************/
__global__ void stencil_shared(float *A, float *B)
{
    __shared__ float in_s[BY + 2][BX + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.y * BY + ty;
    int j = blockIdx.x * BX + tx;

    int si = ty + 1;
    int sj = tx + 1;

    // Load center
    if (i < ROWS && j < COLS)
        in_s[si][sj] = B[IDX(i, j)];
    else
        in_s[si][sj] = 0.0f;

    // Load halos (exactly like textbook)
    if (ty == 0)
        in_s[0][sj] = (i > 0) ? B[IDX(i - 1, j)] : in_s[si][sj];

    if (ty == BY - 1)
        in_s[BY + 1][sj] = (i + 1 < ROWS) ? B[IDX(i + 1, j)] : in_s[si][sj];

    if (tx == 0)
        in_s[si][0] = (j > 0) ? B[IDX(i, j - 1)] : in_s[si][sj];

    if (tx == BX - 1)
        in_s[si][BX + 1] = (j + 1 < COLS) ? B[IDX(i, j + 1)] : in_s[si][sj];

    __syncthreads();

    if (i < ROWS && j < COLS)
    {
        A[IDX(i, j)] =
            in_s[si][sj] +
            in_s[si - 1][sj] +
            in_s[si + 1][sj] +
            in_s[si][sj - 1] +
            in_s[si][sj + 1];
    }
}

/************* CASE 3: TRAPEZOID, GLOBAL MEM ********************/
__global__ void stencil_trap_global(float *A, float *B)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= ROWS || j >= COLS)
        return;

    int top = max(i - 1, 0);
    int bottom = min(i + 1, ROWS - 1);
    int left = max(j - 1, 0);
    int right = min(j + 1, COLS - 1);

    A[IDX(i, j)] =
        B[IDX(i, j)] +
        B[IDX(top, j)] +
        B[IDX(bottom, j)] +
        B[IDX(i, left)] +
        B[IDX(i, right)];
}

/************* CASE 4: TRAPEZOID + SHARED ***********************/
__global__ void stencil_trap_shared(float *A, float *B)
{
    __shared__ float tile[BY + 2][BX + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int i = blockIdx.y * BY + ty;
    int j = blockIdx.x * BX + tx;

    int si = ty + 1;
    int sj = tx + 1;

    // Center
    tile[si][sj] = (i < ROWS && j < COLS) ? B[IDX(i, j)] : 0.0f;

    // Halos
    if (ty == 0)
        tile[0][sj] = (i > 0) ? B[IDX(i - 1, j)] : B[IDX(i, j)];
    if (ty == BY - 1)
        tile[BY + 1][sj] = (i + 1 < ROWS) ? B[IDX(i + 1, j)] : B[IDX(i, j)];
    if (tx == 0)
        tile[si][0] = (j > 0) ? B[IDX(i, j - 1)] : B[IDX(i, j)];
    if (tx == BX - 1)
        tile[si][BX + 1] = (j + 1 < COLS) ? B[IDX(i, j + 1)] : B[IDX(i, j)];

    __syncthreads();

    if (i < ROWS && j < COLS)
    {
        A[IDX(i, j)] =
            tile[si][sj] +
            tile[si - 1][sj] +
            tile[si + 1][sj] +
            tile[si][sj - 1] +
            tile[si][sj + 1];
    }
}


bool check(float *a, float *b)
{
    for (int i = 0; i < ROWS * COLS; i++)
        if (fabs(a[i] - b[i]) > 1e-3)
            return false;
    return true;
}

int main()
{
    size_t N = ROWS * COLS * sizeof(float);

    float *hA = (float *)malloc(N);
    float *hB = (float *)malloc(N);
    float *ref = (float *)malloc(N);

    for (int i = 0; i < ROWS * COLS; i++)
        hB[i] = 1.0f * (i + 1);

    memcpy(ref, hB, N);
    cpu_stencil(hA, ref);

    float *dA, *dB;
    cudaMalloc(&dA, N);
    cudaMalloc(&dB, N);
    cudaMemcpy(dB, hB, N, cudaMemcpyHostToDevice);

    dim3 block(BX, BY);
    dim3 grid((COLS + BX - 1) / BX, (ROWS + BY - 1) / BY);

    cudaEvent_t start, stop;
    float time_ms;

    /************* CASE 1: GLOBAL, NO TRAPEZOID *************/
    cudaMemcpy(dB, hB, N, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < ITERS; t++)
    {
        stencil_global<<<grid, block>>>(dA, dB);
        std::swap(dA, dB);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(hA, dB, N, cudaMemcpyDeviceToHost);
    printf("Case 1 Global | Time: %.3f ms | Correct: %s\n",
           time_ms, check(hA, ref) ? "YES" : "NO");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /************* CASE 2: SHARED, NO TRAPEZOID *************/
    cudaMemcpy(dB, hB, N, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < ITERS; t++)
    {
        stencil_shared<<<grid, block>>>(dA, dB);
        std::swap(dA, dB);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(hA, dB, N, cudaMemcpyDeviceToHost);
    printf("Case 2 Shared | Time: %.3f ms | Correct: %s\n",
           time_ms, check(hA, ref) ? "YES" : "NO");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    /************* CASE 3: TRAPEZOID, GLOBAL *************/
    cudaMemcpy(dB, hB, N, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < ITERS; t++)
    {
        stencil_trap_global<<<grid, block>>>(dA, dB);
        std::swap(dA, dB);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(hA, dB, N, cudaMemcpyDeviceToHost);
    printf("Case 3 Trap Global | Time: %.3f ms | Correct: %s\n",
           time_ms, check(hA, ref) ? "YES" : "NO");

    /************* CASE 4: TRAPEZOID, SHARED *************/
    cudaMemcpy(dB, hB, N, cudaMemcpyHostToDevice);

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    for (int t = 0; t < ITERS; t++)
    {
        stencil_trap_shared<<<grid, block>>>(dA, dB);
        std::swap(dA, dB); 
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(hA, dB, N, cudaMemcpyDeviceToHost);
    printf("Case 4 Trap Shared | Time: %.3f ms | Correct: %s\n",
           time_ms, check(hA, ref) ? "YES" : "NO");

    return 0;
}
