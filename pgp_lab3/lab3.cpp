#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <chrono>

using duration = std::chrono::microseconds;

struct uchar4
{
    unsigned char x = 0;
    unsigned char y = 0;
    unsigned char z = 0;
    unsigned char w = 0;
};

struct double3
{
    double x = 0;
    double y = 0;
    double z = 0;
};

struct int2
{
    int x = 0;
    int y = 0;
};

double3 avg[32];          // выборочное мат ожидание
double logAbsDet[32];     // log(|det(cov)|)
double covInv[3 * 32][3]; // обратная выборочная матрица ковариаций

// дискриминантная функция
double D(uchar4 p, int i)
{
    return -(((p.x - avg[(i / 3)].x) * covInv[i][0] + (p.y - avg[(i / 3)].y) * covInv[i + 1][0] + (p.z - avg[(i / 3)].z) * covInv[i + 2][0]) * (p.x - avg[(i / 3)].x) +
             ((p.x - avg[(i / 3)].x) * covInv[i][1] + (p.y - avg[(i / 3)].y) * covInv[i + 1][1] + (p.z - avg[(i / 3)].z) * covInv[i + 2][1]) * (p.y - avg[(i / 3)].y) +
             ((p.x - avg[(i / 3)].x) * covInv[i][2] + (p.y - avg[(i / 3)].y) * covInv[i + 1][2] + (p.z - avg[(i / 3)].z) * covInv[i + 2][2]) * (p.z - avg[(i / 3)].z)) -
           logAbsDet[(i / 3)];
}

// метод максимального правдоподобия (ММП)
void MLE(uchar4 *data, int w, int h, int nc)
{
    // вычисляем argmax дискриминантной функции
    double max, cur;
    int ic = 0;
    for (int x = 0; x < w * h; x++)
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
    char path_to_in_file[260], path_to_out_file[260];
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
    for (int i = 0; i < nc; i++)
    {
        scanf("%d", &np);
        classes[i].resize(np);

        avg[i].x = 0.0;
        avg[i].y = 0.0;
        avg[i].z = 0.0;

        for (int j = 0; j < np; j++)
        {
            scanf("%d%d", &x, &y);
            classes[i][j].x = x;
            classes[i][j].y = y;

            avg[i].x += (double)data[y * w + x].x;
            avg[i].y += (double)data[y * w + x].y;
            avg[i].z += (double)data[y * w + x].z;
        }

        avg[i].x /= (double)np;
        avg[i].y /= (double)np;
        avg[i].z /= (double)np;
    }

    // матрицы ковариаций
    double cov[3 * nc][3];
    double r, g, b;
    for (int i = 0; i < (3 * nc); i += 3)
    {
        // инициализация суммы
        for (int ch = 0; ch < 3; ch++)
        {
            cov[i + ch][0] = 0.0;
            cov[i + ch][1] = 0.0;
            cov[i + ch][2] = 0.0;
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

            cov[i][0] += (r - avg[(i / 3)].x) * (r - avg[(i / 3)].x);
            cov[i][1] += (r - avg[(i / 3)].x) * (g - avg[(i / 3)].y);
            cov[i][2] += (r - avg[(i / 3)].x) * (b - avg[(i / 3)].z);

            cov[i + 1][0] += (g - avg[(i / 3)].y) * (r - avg[(i / 3)].x);
            cov[i + 1][1] += (g - avg[(i / 3)].y) * (g - avg[(i / 3)].y);
            cov[i + 1][2] += (g - avg[(i / 3)].y) * (b - avg[(i / 3)].z);

            cov[i + 2][0] += (b - avg[(i / 3)].z) * (r - avg[(i / 3)].x);
            cov[i + 2][1] += (b - avg[(i / 3)].z) * (g - avg[(i / 3)].y);
            cov[i + 2][2] += (b - avg[(i / 3)].z) * (b - avg[(i / 3)].z);
        }

        for (int ch = 0; ch < 3; ch++)
        {
            cov[i + ch][0] /= (double)(np - 1);
            cov[i + ch][1] /= (double)(np - 1);
            cov[i + ch][2] /= (double)(np - 1);
        }
    }

    // определитель i-й матрицы ковариаций
    for (int i = 0; i < (3 * nc); i += 3)
    {
        /*
        cov[i][0]     cov[i][1]     cov[i][2]

        cov[i + 1][0] cov[i + 1][1] cov[i + 1][2]

        cov[i + 2][0] cov[i + 2][1] cov[i + 2][2]
        */
        logAbsDet[i / 3] = cov[i][0] * cov[i + 1][1] * cov[i + 2][2] +
                           cov[i][1] * cov[i + 1][2] * cov[i + 2][0] +
                           cov[i][2] * cov[i + 1][0] * cov[i + 2][1] -
                           cov[i][2] * cov[i + 1][1] * cov[i + 2][0] -
                           cov[i][1] * cov[i + 1][0] * cov[i + 2][2] -
                           cov[i][0] * cov[i + 1][2] * cov[i + 2][1];
    }

    // обратные матрицы ковариаций
    for (int i = 0; i < (3 * nc); i += 3)
    {
        covInv[i][0] = cov[i + 1][1] * cov[i + 2][2] - cov[i + 1][2] * cov[i + 2][1];
        covInv[i][1] = -(cov[i + 1][0] * cov[i + 2][2] - cov[i + 1][2] * cov[i + 2][0]);
        covInv[i][2] = cov[i + 1][0] * cov[i + 2][1] - cov[i + 1][1] * cov[i + 2][0];

        covInv[i + 1][0] = -(cov[i][1] * cov[i + 2][2] - cov[i][2] * cov[i + 2][1]);
        covInv[i + 1][1] = cov[i][0] * cov[i + 2][2] - cov[i][2] * cov[i + 2][0];
        covInv[i + 1][2] = -(cov[i][0] * cov[i + 2][1] - cov[i][1] * cov[i + 2][0]);

        covInv[i + 2][0] = cov[i][1] * cov[i + 1][2] - cov[i][2] * cov[i + 1][1];
        covInv[i + 2][1] = -(cov[i][0] * cov[i + 1][2] - cov[i][2] * cov[i + 1][0]);
        covInv[i + 2][2] = cov[i][0] * cov[i + 1][1] - cov[i][1] * cov[i + 1][0];

        for (int ch = 0; ch < 3; ch++)
        {
            covInv[i + ch][0] /= logAbsDet[(i / 3)];
            covInv[i + ch][1] /= logAbsDet[(i / 3)];
            covInv[i + ch][2] /= logAbsDet[(i / 3)];
        }

        logAbsDet[(i / 3)] = std::log(std::abs(logAbsDet[(i / 3)]));
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
    float t = 0;
    start = std::chrono::high_resolution_clock::now();

    MLE(data, w, h, nc);

    stop = std::chrono::high_resolution_clock::now();
    t += std::chrono::duration_cast<duration>(stop - start).count();
    std::cout << "time = " << t << " ms" << std::endl;

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