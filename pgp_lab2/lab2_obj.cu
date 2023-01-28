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

__device__ float modeClamp(int p, int b)
{
    return (float)max(min(p, b), 0);
}

__global__ void gaussianBlurX(uchar4 *out, cudaTextureObject_t texObj, int w, int h, int r)
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
                p = tex2D<uchar4>(texObj, modeClamp(x + i, w), modeClamp(y, h));
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];

                p = tex2D<uchar4>(texObj, modeClamp(x - i, w), modeClamp(y, h));
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];
            }
            p = tex2D<uchar4>(texObj, x, y);
            f.x += p.x * kernel[0];
            f.y += p.y * kernel[0];
            f.z += p.z * kernel[0];

            out[y * w + x] = make_uchar4((unsigned char)f.x, (unsigned char)f.y, (unsigned char)f.z, p.w);
        }
    }
}

__global__ void gaussianBlurY(uchar4 *out, cudaTextureObject_t texObj, int w, int h, int r)
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
                p = tex2D<uchar4>(texObj, modeClamp(x, w), modeClamp((y + i), h));
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];

                p = tex2D<uchar4>(texObj, modeClamp(x, w), modeClamp((y - i), h));
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];
            }
            p = tex2D<uchar4>(texObj, x, y);
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
        // Выделяем CUDA массив в памяти GPU
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uchar4>();
        cudaArray_t cuArray;
        CSC(cudaMallocArray(&cuArray, &channelDesc, w, h));

        // Установим pitch (кол-во байт в одной строке наших данных, включая выравнивание)
        const size_t spitch = w * sizeof(uchar4); // мы не используем выравнивание
        // Копируем данные из CPU в память GPU
        CSC(cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

        // Указываем текстуру
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray;

        // Задаем параметры объекта текстуры
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.addressMode[0] = cudaAddressModeClamp;  // дублируем граничные значения при выходе за границу по x
        texDesc.addressMode[1] = cudaAddressModeClamp;  // дублируем граничные значения при выходе за границу по y
        texDesc.filterMode = cudaFilterModePoint;       // не используем билинейную интерполяцию (просто ближайший пиксель)
        texDesc.readMode = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;                   // координаты [0, w-1] x [0, h-1]

        // Создаем текстурный объект
        cudaTextureObject_t texObj = 0;
        CSC(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

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

        // Вызваем ядро
        dim3 threadsperBlock(32, 32);
        dim3 numBlocks(16, 16);
        gaussianBlurX<<<numBlocks, threadsperBlock>>>(dev_out, texObj, w, h, r);
        CSC(cudaGetLastError());

        // Копируем данные с GPU обратно на CPU
        CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));
        CSC(cudaMemcpy2DToArray(cuArray, 0, 0, data, spitch, w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

        // Вызваем ядро
        gaussianBlurY<<<numBlocks, threadsperBlock>>>(dev_out, texObj, w, h, r);
        CSC(cudaGetLastError());

        // Копируем данные с GPU обратно на CPU
        CSC(cudaMemcpy(data, dev_out, sizeof(uchar4) * w * h, cudaMemcpyDeviceToHost));

        // Уничтожаем объект текстуры
        CSC(cudaDestroyTextureObject(texObj));

        // Освобождаем память GPU
        CSC(cudaFreeArray(cuArray));
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