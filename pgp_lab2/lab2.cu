#include <stdio.h>
#include <stdlib.h>

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

__constant__ float kernel[1024]; // ядро свертки
texture<uchar4, 2, cudaReadModeElementType> tex;

__device__ float modeClamp(int p, int b)
{
    return (float)max(min(p, b), 0);
}

__global__ void gaussianBlurX(uchar4 *out, int w, int h, int r)
{
    // вычисляем абсолютный номер потока
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    // вычисляем число потоков - это будет наш шаг, если потоков меньше чем (h, w)
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    uchar4 p; // пиксель
    float4 f; // отклик фильтрации
    for (int y = idy; y < h; y += offsety)
    {
        for (int x = idx; x < w; x += offsetx)
        {
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
            for (int i = 1; i <= r; ++i)
            {
                p = tex2D(tex, modeClamp(x + i, w), modeClamp(y, h));
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];

                p = tex2D(tex, modeClamp(x - i, w), modeClamp(y, h));
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];
            }
            p = tex2D(tex, x, y);
            f.x += p.x * kernel[0];
            f.y += p.y * kernel[0];
            f.z += p.z * kernel[0];

            out[y * w + x] = make_uchar4((unsigned char)f.x, (unsigned char)f.y, (unsigned char)f.z, p.w);
        }
    }
}

__global__ void gaussianBlurY(uchar4 *out, int w, int h, int r)
{
    // вычисляем абсолютный номер потока
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;

    // вычисляем число потоков - это будет наш шаг, если потоков меньше чем (h, w)
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;

    uchar4 p; // пиксель
    float4 f; // отклик фильтрации
    for (int y = idy; y < h; y += offsety)
    {
        for (int x = idx; x < w; x += offsetx)
        {
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
            for (int i = 1; i <= r; ++i)
            {
                p = tex2D(tex, modeClamp(x, w), modeClamp((y + i), h));
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];

                p = tex2D(tex, modeClamp(x, w), modeClamp((y - i), h));
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];
            }
            p = tex2D(tex, x, y);
            f.x += p.x * kernel[0];
            f.y += p.y * kernel[0];
            f.z += p.z * kernel[0];

            out[y * w + x] = make_uchar4((unsigned char)f.x, (unsigned char)f.y, (unsigned char)f.z, p.w);
        }
    }
}

int main()
{
    // Считываем входные данные
    char path_to_in_file[260], path_to_out_file[260];
    int r;
    scanf("%s", path_to_in_file);
    scanf("%s", path_to_out_file);
    scanf("%d", &r);

    // Чтение из файла
    FILE *fp = fopen(path_to_in_file, "rb");
    int w, h;
    fread(&w, sizeof(int), 1, fp);
    fread(&h, sizeof(int), 1, fp);
    uchar4 *data = (uchar4 *)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    if (r > 0)
    {
        cudaArray *arr;
        cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
        CSC(cudaMallocArray(&arr, &ch, w, h));

        CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

        tex.normalized = false;
        tex.filterMode = cudaFilterModePoint;
        tex.channelDesc = ch;
        tex.addressMode[0] = cudaAddressModeClamp;
        tex.addressMode[1] = cudaAddressModeClamp;

        CSC(cudaBindTextureToArray(tex, arr, ch));

        // Размещаем результат преобразования в памяти GPU
        uchar4 *dev_out;
        CSC(cudaMalloc(&dev_out, sizeof(uchar4) * w * h));

        // Инициализируем фильтр размытия по Гауссу
        float hostKernel[1024];
        float sum = 0;
        for (int i = 0; i <= r; i++)
        {
            hostKernel[i] = exp((float)(-i * i) / (float)(2 * r * r));
            sum += 2 * hostKernel[i];
        }
        sum -= hostKernel[0];
        for (int i = 0; i <= r; i++)
        {
            hostKernel[i] /= sum;
        }
        CSC(cudaMemcpyToSymbol(kernel, hostKernel, sizeof(hostKernel), 0, cudaMemcpyHostToDevice));

        cudaEvent_t start, stop;
        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start));

        // Вызваем ядро
        dim3 threadsperBlock(32, 32);
        dim3 numBlocks(1024, 1024);
        gaussianBlurX<<<numBlocks, threadsperBlock>>>(dev_out, w, h, r);
        CSC(cudaGetLastError());

        CSC(cudaEventRecord(stop));
        CSC(cudaEventSynchronize(stop));

        float t1;
        CSC(cudaEventElapsedTime(&t1, start, stop));
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));

        // Копируем данные с GPU обратно на CPU
        CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

        CSC(cudaEventCreate(&start));
        CSC(cudaEventCreate(&stop));
        CSC(cudaEventRecord(start));

        // Вызваем ядро
        gaussianBlurY<<<numBlocks, threadsperBlock>>>(dev_out, w, h, r);
        CSC(cudaGetLastError());

        CSC(cudaEventRecord(stop));
        CSC(cudaEventSynchronize(stop));

        float t2;
        CSC(cudaEventElapsedTime(&t2, start, stop));
        CSC(cudaEventDestroy(start));
        CSC(cudaEventDestroy(stop));

        float t = t1 + t2;
        printf("time = %f ms\n", t);

        // Копируем данные с GPU обратно на CPU
        CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

        // Уничтожаем объект текстуры
        CSC(cudaUnbindTexture(tex));

        // Освобождаем память GPU
        CSC(cudaFreeArray(arr));
        CSC(cudaFree(dev_out));
    }

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