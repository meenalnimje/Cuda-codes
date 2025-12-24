#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#define ROWS 1024
#define COLS 1024
#define ITER 100
#define BLOCK 16
#define EPS 1e-4
#define T 4
#define IDX(i, j) ((i) * COLS + (j))

// Trapezoid method
#define H 3 // trapezoid height
#define TILE 16
#define SHARED (TILE + 2 * H)

// verification for Case 1 and Case 2
bool verify(float *a, float *b)
{
    for (int i = 0; i < ROWS * COLS; i++)
        if (fabs(a[i] - b[i]) > EPS)
            return false;
    return true;
}

// verification for Case 3 and Case 4
bool verify_interior(float *a, float *b)
{
    for (int i = H; i < ROWS - H; i++)
        for (int j = H; j < COLS - H; j++)
            if (fabs(a[IDX(i, j)] - b[IDX(i, j)]) > EPS)
                return false;
    return true;
}

// CPU reference
void cpu_stencil(float *A, float *B)
{
    for (int k = 0; k < ITER; k++)
    {
        for (int i = 0; i < ROWS; i++)
        {
            for (int j = 0; j < COLS; j++)
            {
                int top = (i == 0) ? 0 : i - 1;
                int bottom = (i == ROWS - 1) ? ROWS - 1 : i + 1;
                int left = (j == 0) ? 0 : j - 1;
                int right = (j == COLS - 1) ? COLS - 1 : j + 1;

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

// Case:1  Baseline – Global Memory Only

__global__ void stencil_global(float *A, float *B)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

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

// Case 2: Baseline - Shared Memory

__global__ void stencil_shared(float *A, float *B)
{
    __shared__ float tile[BLOCK + 2][BLOCK + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int i = blockIdx.y * BLOCK + ty;
    int j = blockIdx.x * BLOCK + tx;

    int si = ty + 1;
    int sj = tx + 1;

    if (i < ROWS && j < COLS)
        tile[si][sj] = B[IDX(i, j)];

    if (ty == 0 && i > 0)
        tile[0][sj] = B[IDX(i - 1, j)];
    if (ty == BLOCK - 1 && i < ROWS - 1)
        tile[si + 1][sj] = B[IDX(i + 1, j)];
    if (tx == 0 && j > 0)
        tile[si][0] = B[IDX(i, j - 1)];
    if (tx == BLOCK - 1 && j < COLS - 1)
        tile[si][sj + 1] = B[IDX(i, j + 1)];

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

// case 3: TRAPEZOID - GLOBAL MEMORY

__global__ void stencil_true_trap_global(float *A, float *B)
{
    int i = blockIdx.y * TILE + threadIdx.y;
    int j = blockIdx.x * TILE + threadIdx.x;

    if (threadIdx.x >= TILE || threadIdx.y >= TILE)
        return;
    if (i < H || i >= ROWS - H || j < H || j >= COLS - H)
        return;

    float curr = B[IDX(i, j)];

    for (int t = 0; t < H; t++)
    {
        int top = i - 1;
        int bottom = i + 1;
        int left = j - 1;
        int right = j + 1;

        curr =
            curr +
            B[IDX(top, j)] +
            B[IDX(bottom, j)] +
            B[IDX(i, left)] +
            B[IDX(i, right)];

        __syncthreads();
    }

    A[IDX(i, j)] = curr;
}

// Case 4:TRAPEZOID - SHARED MEMORY
__global__ void stencil_true_trap_shared(float *A, float *B)
{
    __shared__ float tile[SHARED][SHARED];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int bx = blockIdx.x * TILE;
    int by = blockIdx.y * TILE;

    int gi = by + ty - H;
    int gj = bx + tx - H;

    // Load halo
    if (gi >= 0 && gi < ROWS && gj >= 0 && gj < COLS)
        tile[ty][tx] = B[IDX(gi, gj)];
    else
        tile[ty][tx] = 0.0f;

    __syncthreads();

    // Trapezoid time steps
    for (int t = 0; t < H; t++)
    {
        if (ty >= H + t && ty < SHARED - H - t &&
            tx >= H + t && tx < SHARED - H - t)
        {
            tile[ty][tx] =
                tile[ty][tx] +
                tile[ty - 1][tx] +
                tile[ty + 1][tx] +
                tile[ty][tx - 1] +
                tile[ty][tx + 1];
        }
        __syncthreads();
    }

    if (ty >= H && ty < H + TILE &&
        tx >= H && tx < H + TILE)
    {
        int oi = by + ty - H;
        int oj = bx + tx - H;

        if (oi >= H && oi < ROWS - H &&
            oj >= H && oj < COLS - H)
        {
            A[IDX(oi, oj)] = tile[ty][tx];
        }
    }
}

int main()
{
    size_t size = ROWS * COLS * sizeof(float);

    // Host memory
    float *hA = (float *)malloc(size);
    float *hB = (float *)malloc(size);
    float *hRef = (float *)malloc(size);
    float *hOut = (float *)malloc(size);

    // Initialize input
    for (int i = 0; i < ROWS * COLS; i++)
        hB[i] = (float)(i + 1);

    // ---------- CPU Reference ----------
    memcpy(hRef, hB, size);
    cpu_stencil(hA, hRef); // hRef holds final CPU result

    // Device memory
    float *dA, *dB;
    cudaMalloc(&dA, size);
    cudaMalloc(&dB, size);

    dim3 block(BLOCK, BLOCK);
    dim3 grid((COLS + BLOCK - 1) / BLOCK,
              (ROWS + BLOCK - 1) / BLOCK);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time_ms;
    bool ok;

    // ===================== CASE 1 =====================

    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int k = 0; k < ITER; k++)
    {
        stencil_global<<<grid, block>>>(dA, dB);
        std::swap(dA, dB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(hOut, dB, size, cudaMemcpyDeviceToHost);
    ok = verify(hOut, hRef);
    printf("Case 1 (Baseline Global)        : %s | Time = %.3f ms\n",
           ok ? "YES" : "NO", time_ms);

    // ===================== CASE 2 =====================

    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int k = 0; k < ITER; k++)
    {
        stencil_shared<<<grid, block>>>(dA, dB);
        std::swap(dA, dB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(hOut, dB, size, cudaMemcpyDeviceToHost);
    ok = verify(hOut, hRef);
    printf("Case 2 (Shared Memory)          : %s | Time = %.3f ms\n",
           ok ? "YES" : "NO", time_ms);

    // ===================== CASE 3 =====================

    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int k = 0; k < ITER; k += H)
    {
        stencil_true_trap_global<<<grid, block>>>(dA, dB);
        std::swap(dA, dB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(hOut, dB, size, cudaMemcpyDeviceToHost);
    ok = verify_interior(hOut, hRef);
    printf("Case 3 (Trapezoid Global Memory)  : %s | Time = %.3f ms\n",
           ok ? "YES" : "NO", time_ms);

    // ===================== CASE 4 =====================

    dim3 block_trap(SHARED, SHARED);
    dim3 grid_trap((COLS + TILE - 1) / TILE,
                   (ROWS + TILE - 1) / TILE);

    cudaMemcpy(dB, hB, size, cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    for (int k = 0; k < ITER; k += H)
    {
        stencil_true_trap_shared<<<grid_trap, block_trap>>>(dA, dB);
        std::swap(dA, dB);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);

    cudaMemcpy(hOut, dB, size, cudaMemcpyDeviceToHost);
    ok = verify_interior(hOut, hRef);
    printf("Case 4 (Trapezoid Shared Memory)  : %s | Time = %.3f ms\n",
           ok ? "YES" : "NO", time_ms);

    // Cleanup
    cudaFree(dA);
    cudaFree(dB);
    free(hA);
    free(hB);
    free(hRef);
    free(hOut);

    return 0;
}
