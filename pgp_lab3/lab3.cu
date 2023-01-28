#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>

#define CSC(call)                                                 \
    do                                                            \
    {                                                             \
        cudaError_t res = call;                                   \
        if (res != cudaSuccess)                                   \
        {                                                         \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n",      \
                    __FILE__, __LINE__, cudaGetErrorString(res)); \
            exit(0);                                              \
        }                                                         \
    } while (0)

__constant__ double3 avg[32];          // выборочное мат ожидание
__constant__ double logAbsDet[32];     // log(|det(cov)|)
__constant__ double covInv[3 * 32][3]; // обратная выборочная матрица ковариаций

// дискриминантная функция
__device__ double D(uchar4 p, int i)
{
    return -(((p.x - avg[(i / 3)].x) * covInv[i][0] + (p.y - avg[(i / 3)].y) * covInv[i + 1][0] + (p.z - avg[(i / 3)].z) * covInv[i + 2][0]) * (p.x - avg[(i / 3)].x) +
             ((p.x - avg[(i / 3)].x) * covInv[i][1] + (p.y - avg[(i / 3)].y) * covInv[i + 1][1] + (p.z - avg[(i / 3)].z) * covInv[i + 2][1]) * (p.y - avg[(i / 3)].y) +
             ((p.x - avg[(i / 3)].x) * covInv[i][2] + (p.y - avg[(i / 3)].y) * covInv[i + 1][2] + (p.z - avg[(i / 3)].z) * covInv[i + 2][2]) * (p.z - avg[(i / 3)].z)) -
           logAbsDet[(i / 3)];
}

// метод максимального правдоподобия (ММП)
__global__ void MLE(uchar4 *data, int w, int h, int nc)
{
    // вычисляем абсолютный номер потока
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // вычисляем число потоков - это будет наш шаг, если потоков меньше чем n
    int offset = blockDim.x * gridDim.x;

    // вычисляем argmax дискриминантной функции
    double max, cur;
    int ic;
    for (int x = idx; x < w * h; x += offset)
    {
        max = -DBL_MAX;
        for (int i = 0; i < (3 * nc); i += 3)
        {
            cur = D(data[x], i);
            if (cur > max)
            {
                max = cur;
                ic = (i / 3);
            }
        }
        data[x].w = (unsigned char)ic;
    }
}

int main()
{
    // Считываем входные данные
    char path_to_in_file[_POSIX_PATH_MAX], path_to_out_file[_POSIX_PATH_MAX];
    scanf("%s", path_to_in_file);
    scanf("%s", path_to_out_file);

    // Чтение из файла
    FILE *fp = fopen(path_to_in_file, "rb");
    int w, h;
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    int nc; // число классов
    scanf("%d", &nc);

    int np; // число пикселей i-й выборки
    int x, y;
    std::vector<std::vector<int2>> classes(nc);

    // вектор средних для i-й выборки из np пикселей
    double3 hostAvg[nc];
    for (int i = 0; i < nc; i++)
    {
        scanf("%d", &np);
        classes[i].resize(np);

        hostAvg[i].x = 0.0;
        hostAvg[i].y = 0.0;
        hostAvg[i].z = 0.0;

        for (int j = 0; j < np; j++)
        {
            scanf("%d%d", &x, &y);
            classes[i][j] = make_int2(x, y);

            hostAvg[i].x += (double)data[y * w + x].x;
            hostAvg[i].y += (double)data[y * w + x].y;
            hostAvg[i].z += (double)data[y * w + x].z;
        }

        hostAvg[i].x /= (double)np;
        hostAvg[i].y /= (double)np;
        hostAvg[i].z /= (double)np;
    }

    // матрицы ковариаций
    double hostCov[3 * nc][3];
    double r, g, b;
    for (int i = 0; i < (3 * nc); i += 3)
    {
        // инициализация суммы
        for (int ch = 0; ch < 3; ch++)
        {
            hostCov[i + ch][0] = 0.0;
            hostCov[i + ch][1] = 0.0;
            hostCov[i + ch][2] = 0.0;
        }

        np = classes[(i / 3)].size();
        for (size_t j = 0; j < np; j++)
        {
            x = classes[(i / 3)][j].x;
            y = classes[(i / 3)][j].y;

            r = data[y * w + x].x;
            g = data[y * w + x].y;
            b = data[y * w + x].z;

            /*
            (r - avg.r)                                             (r - avg.r) * (r - avg.r)  (r - avg.r) * (g - avg.g)  (r - avg.r) * (b - avg.b)
            (g - avg.g)  *  (r - avg.r) (g - avg.g) (b - avg.b)  =  (g - avg.g) * (r - avg.r)  (g - avg.g) * (g - avg.g)  (g - avg.g) * (b - avg.b)
            (b - avg.b)                                             (b - avg.b) * (r - avg.r)  (b - avg.b) * (g - avg.g)  (b - avg.b) * (b - avg.b)
            */

            hostCov[i][0] += (r - hostAvg[(i / 3)].x) * (r - hostAvg[(i / 3)].x);
            hostCov[i][1] += (r - hostAvg[(i / 3)].x) * (g - hostAvg[(i / 3)].y);
            hostCov[i][2] += (r - hostAvg[(i / 3)].x) * (b - hostAvg[(i / 3)].z);

            hostCov[i + 1][0] += (g - hostAvg[(i / 3)].y) * (r - hostAvg[(i / 3)].x);
            hostCov[i + 1][1] += (g - hostAvg[(i / 3)].y) * (g - hostAvg[(i / 3)].y);
            hostCov[i + 1][2] += (g - hostAvg[(i / 3)].y) * (b - hostAvg[(i / 3)].z);

            hostCov[i + 2][0] += (b - hostAvg[(i / 3)].z) * (r - hostAvg[(i / 3)].x);
            hostCov[i + 2][1] += (b - hostAvg[(i / 3)].z) * (g - hostAvg[(i / 3)].y);
            hostCov[i + 2][2] += (b - hostAvg[(i / 3)].z) * (b - hostAvg[(i / 3)].z);
        }

        for (int ch = 0; ch < 3; ch++)
        {
            hostCov[i + ch][0] /= (double)(np - 1);
            hostCov[i + ch][1] /= (double)(np - 1);
            hostCov[i + ch][2] /= (double)(np - 1);
        }
    }

    // определитель i-й матрицы ковариаций
    double hostDet[nc];
    for (int i = 0; i < (3 * nc); i += 3)
    {
        /*
        hostCov[i][0]     hostCov[i][1]     hostCov[i][2]

        hostCov[i + 1][0] hostCov[i + 1][1] hostCov[i + 1][2]

        hostCov[i + 2][0] hostCov[i + 2][1] hostCov[i + 2][2]
        */
        hostDet[i / 3] = hostCov[i][0] * hostCov[i + 1][1] * hostCov[i + 2][2] +
                         hostCov[i][1] * hostCov[i + 1][2] * hostCov[i + 2][0] +
                         hostCov[i][2] * hostCov[i + 1][0] * hostCov[i + 2][1] -
                         hostCov[i][2] * hostCov[i + 1][1] * hostCov[i + 2][0] -
                         hostCov[i][1] * hostCov[i + 1][0] * hostCov[i + 2][2] -
                         hostCov[i][0] * hostCov[i + 1][2] * hostCov[i + 2][1];
    }

    // обратные матрицы ковариаций
    double hostCovInv[3 * nc][3];
    for (int i = 0; i < (3 * nc); i += 3)
    {
        hostCovInv[i][0] = hostCov[i + 1][1] * hostCov[i + 2][2] - hostCov[i + 1][2] * hostCov[i + 2][1];
        hostCovInv[i][1] = -(hostCov[i + 1][0] * hostCov[i + 2][2] - hostCov[i + 1][2] * hostCov[i + 2][0]);
        hostCovInv[i][2] = hostCov[i + 1][0] * hostCov[i + 2][1] - hostCov[i + 1][1] * hostCov[i + 2][0];

        hostCovInv[i + 1][0] = -(hostCov[i][1] * hostCov[i + 2][2] - hostCov[i][2] * hostCov[i + 2][1]);
        hostCovInv[i + 1][1] = hostCov[i][0] * hostCov[i + 2][2] - hostCov[i][2] * hostCov[i + 2][0];
        hostCovInv[i + 1][2] = -(hostCov[i][0] * hostCov[i + 2][1] - hostCov[i][1] * hostCov[i + 2][0]);

        hostCovInv[i + 2][0] = hostCov[i][1] * hostCov[i + 1][2] - hostCov[i][2] * hostCov[i + 1][1];
        hostCovInv[i + 2][1] = -(hostCov[i][0] * hostCov[i + 1][2] - hostCov[i][2] * hostCov[i + 1][0]);
        hostCovInv[i + 2][2] = hostCov[i][0] * hostCov[i + 1][1] - hostCov[i][1] * hostCov[i + 1][0];

        for (int ch = 0; ch < 3; ch++)
        {
            hostCovInv[i + ch][0] /= hostDet[(i / 3)];
            hostCovInv[i + ch][1] /= hostDet[(i / 3)];
            hostCovInv[i + ch][2] /= hostDet[(i / 3)];
        }

        hostDet[(i / 3)] = log(abs(hostDet[(i / 3)]));
    }

    CSC(cudaMemcpyToSymbol(avg, hostAvg, sizeof(hostAvg), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(logAbsDet, hostDet, sizeof(hostDet), 0, cudaMemcpyHostToDevice));
    CSC(cudaMemcpyToSymbol(covInv, hostCovInv, sizeof(hostCovInv), 0, cudaMemcpyHostToDevice));

    // Размещаем результат преобразования в памяти GPU
    uchar4 *dev_data;
    CSC(cudaMalloc(&dev_data, sizeof(uchar4) * w * h));
    CSC(cudaMemcpy(dev_data, data, sizeof(uchar4) * w * h, cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

    // Вызываем ядро
    dim3 threadsperBlock(256);
    dim3 numBlocks(256);
    MLE<<<numBlocks, threadsperBlock>>>(dev_data, w, h, nc);
    CSC(cudaGetLastError());

    CSC(cudaEventRecord(stop));
    CSC(cudaEventSynchronize(stop));

    float t;
    CSC(cudaEventElapsedTime(&t, start, stop));
    CSC(cudaEventDestroy(start));
    CSC(cudaEventDestroy(stop));

    printf("time = %f ms\n", t);

    // Копируем данные с GPU обратно на CPU
    CSC(cudaMemcpy(data, dev_data, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

    // Освобождаем память GPU
    CSC(cudaFree(dev_data));

    // Запись в файл
    fp = fopen(path_to_out_file, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    // Освобождаем память СPU
    free(data);
    return 0;
}