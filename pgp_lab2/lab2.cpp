#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <vector>
#include <chrono>
#include <iostream>

using duration = std::chrono::microseconds;

struct uchar4
{
    unsigned char x = 0;
    unsigned char y = 0;
    unsigned char z = 0;
    unsigned char w = 0;
};

struct float4
{
    float x = 0;
    float y = 0;
    float z = 0;
    float w = 0;
};

float kernel[1024];

int modeClamp(int p, int b)
{
    return std::max(std::min(p, b), 0);
}

void gaussianBlurX(uchar4 *data, int w, int h, int r)
{
    uchar4 p; // пиксель
    float4 f; // отклик фильтрации
    for (int x = 0; x < w; x++)
    {
        for (int y = 0; y < h; y++)
        {
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
            for (int i = 1; i <= r; ++i)
            {
                p = data[modeClamp(x + i, w) + w * modeClamp(y, h)];
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];

                p = data[modeClamp(x - i, w) + w * modeClamp(y, h)];
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];
            }
            p = data[x + w * y];
            f.x += p.x * kernel[0];
            f.y += p.y * kernel[0];
            f.z += p.z * kernel[0];

            data[y * w + x].x = (unsigned char)f.x;
            data[y * w + x].y = (unsigned char)f.y;
            data[y * w + x].z = (unsigned char)f.z;
            data[y * w + x].w = p.w;
        }
    }
}

void gaussianBlurY(uchar4 *data, int w, int h, int r)
{
    uchar4 p; // пиксель
    float4 f; // отклик фильтрации
    for (int x = 0; x < w; x++)
    {
        for (int y = 0; y < h; y++)
        {
            f.x = 0.0;
            f.y = 0.0;
            f.z = 0.0;
            for (int i = 1; i <= r; ++i)
            {
                p = data[modeClamp(x, w) + w * modeClamp((y + i), h)];
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];

                p = data[modeClamp(x, w) + w * modeClamp((y - i), h)];
                f.x += p.x * kernel[i];
                f.y += p.y * kernel[i];
                f.z += p.z * kernel[i];
            }
            p = data[x + w * y];
            f.x += p.x * kernel[0];
            f.y += p.y * kernel[0];
            f.z += p.z * kernel[0];

            data[y * w + x].x = (unsigned char)f.x;
            data[y * w + x].y = (unsigned char)f.y;
            data[y * w + x].z = (unsigned char)f.z;
            data[y * w + x].w = p.w;
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
        // Инициализируем фильтр размытия по Гауссу
        float sum = 0;
        for (int i = 0; i <= r; i++)
        {
            kernel[i] = std::exp((float)(-i * i) / (float)(2 * r * r));
            sum += 2 * kernel[i];
        }
        sum -= kernel[0];
        for (int i = 0; i <= r; i++)
        {
            kernel[i] /= sum;
        }

        std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
        float t = 0;
        start = std::chrono::high_resolution_clock::now();

        gaussianBlurX(data, w, h, r);
        gaussianBlurY(data, w, h, r);
        
        stop = std::chrono::high_resolution_clock::now();
        t += std::chrono::duration_cast<duration>(stop - start).count();
        std::cout << "time = " << t << " ms" << std::endl;
    }

    // Запись в файл
    fp = fopen(path_to_out_file, "wb");
    fwrite(&w, sizeof(int), 1, fp);
    fwrite(&h, sizeof(int), 1, fp);
    fwrite(data, sizeof(uchar4), w * h, fp);
    fclose(fp);

    free(data);
    return 0;
}