#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#define CSC(call)                                                                                             \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t status = call;                                                                            \
        if (status != cudaSuccess)                                                                            \
        {                                                                                                     \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", __FILE__, __LINE__, cudaGetErrorString(status)); \
            exit(0);                                                                                          \
        }                                                                                                     \
    } while (0)

// идентификатор '__global__' говрит, что функция будет работать на gpu
__global__ void parallel_reverse(double *vec, double *vec_reverse, int n)
{
    // вычисляем абсолютный номер потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // вычисляем число потоков - это будет наш шаг, если потоков меньше чем n
    int offset = blockDim.x * gridDim.x;

    while (idx < n)
    {
        assert(idx < n);
        vec_reverse[idx] = vec[n - 1 - idx];
        idx += offset;
    }
}

int main()
{
    int n;
    scanf("%d", &n);

    double *vec = (double *)malloc(sizeof(double) * n);
    for (int i = 0; i < n; i++)
    {
        scanf("%lf", &vec[i]);
    }

    double *dev_vec;
    double *dev_vec_reverse;
    // выделяем память на gpu
    CSC(cudaMalloc(&dev_vec, sizeof(double) * n));
    CSC(cudaMalloc(&dev_vec_reverse, sizeof(double) * n));
    CSC(cudaMemcpy(dev_vec, vec, sizeof(double) * n, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

    // <<<кол-во блоков, размер одного блока>>>
    // max -- <<<65535, 1024>>>
    parallel_reverse<<<256, 256>>>(dev_vec, dev_vec_reverse, n);
    CSC(cudaDeviceSynchronize());
    CSC(cudaGetLastError());

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));

    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    printf("time = %f ms\n", t);

    CSC(cudaMemcpy(vec, dev_vec_reverse, sizeof(double) * n, cudaMemcpyDeviceToHost));

    // for (int i = 0; i < n; i++)
    // {
    //     printf("%.10e ", vec[i]);
    // }
    // printf("\n");

    CSC(cudaFree(dev_vec_reverse));
    CSC(cudaFree(dev_vec));

    free(vec);
    return 0;
}