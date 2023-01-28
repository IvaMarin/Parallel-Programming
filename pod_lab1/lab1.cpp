#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#include <cmath>
#include <iostream>
#include <chrono>

using duration = std::chrono::microseconds;

#define MPI_CALL(call)                                       \
    do                                                       \
    {                                                        \
        int err = (call);                                    \
        if (err != MPI_SUCCESS)                              \
        {                                                    \
            char estring[MPI_MAX_ERROR_STRING];              \
            int len;                                         \
            MPI_Error_string(err, estring, &len);            \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
                    __FILE__, __LINE__, estring);            \
            MPI_Finalize();                                  \
            exit(0);                                         \
        }                                                    \
    } while (false)

// Индексация внутри блока
#define _i(i, j, k) (((k) + 1) * (ny + 2) * (nx + 2) + ((j) + 1) * (nx + 2) + (i) + 1)
#define _iz(id) (((id) / (nx + 2) / (ny + 2)) - 1)
#define _iy(id) ((((id) % ((nx + 2) * (ny + 2))) / (nx + 2)) - 1)
#define _ix(id) ((id) % (nx + 2) - 1)

// Индексация по блокам (процессам)
#define _ib(i, j, k) ((k)*nby * nbx + (j)*nbx + (i))
#define _ibz(id) ((id) / nbx / nby)
#define _iby(id) (((id) % (nbx * nby)) / nbx)
#define _ibx(id) ((id) % nbx)

const int ROOT_RANK = 0;

int main(int argc, char *argv[])
{
    MPI_Status status;
    MPI_CALL(MPI_Init(&argc, &argv));

    // номер процесса, кол-во процесов
    int id, numproc;

    // Получаем номер процесса, кол-во процесов
    MPI_CALL(MPI_Comm_rank(MPI_COMM_WORLD, &id));
    MPI_CALL(MPI_Comm_size(MPI_COMM_WORLD, &numproc));

    // Инициализация параметров расчета
    int nx, ny, nz, nbx, nby, nbz;
    double lx, ly, lz, u_bottom, u_top, u_left, u_right, u_front, u_back, eps, u_0;
    char filename[256];

    std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
	float t = 0;

    if (id == ROOT_RANK)
    {
        // размер сетки блоков (процессов) по одному измерению
        scanf("%d", &nbx);
        scanf("%d", &nby);
        scanf("%d", &nbz);

        // размер блока по одному измерению
        scanf("%d", &nx);
        scanf("%d", &ny);
        scanf("%d", &nz);

        scanf("%s", filename);

        scanf("%lf", &eps);
        scanf("%lf", &lx);
        scanf("%lf", &ly);
        scanf("%lf", &lz);

        scanf("%lf", &u_bottom);
        scanf("%lf", &u_top);
        scanf("%lf", &u_left);
        scanf("%lf", &u_right);
        scanf("%lf", &u_front);
        scanf("%lf", &u_back);

        scanf("%lf", &u_0);

        start = std::chrono::high_resolution_clock::now();
    }

    // Передача параметров расчета всем процессам
    MPI_CALL(MPI_Bcast(&nx, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&ny, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&nz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&nbx, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&nby, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&nbz, 1, MPI_INT, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&lx, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&ly, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&lz, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&eps, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&u_top, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&u_bottom, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&u_left, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&u_right, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&u_front, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&u_back, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));
    MPI_CALL(MPI_Bcast(&u_0, 1, MPI_DOUBLE, ROOT_RANK, MPI_COMM_WORLD));

    // Вычисляем индексы в трехмерной сетке
    int ib, jb, kb;
    ib = _ibx(id);
    jb = _iby(id);
    kb = _ibz(id);

    double hx, hy, hz;
    hx = lx / (double)(nx * nbx);
    hy = ly / (double)(ny * nby);
    hz = lz / (double)(nz * nbz);

    double h2x, h2y, h2z;
    h2x = 1.0 / (double)(hx * hx);
    h2y = 1.0 / (double)(hy * hy);
    h2z = 1.0 / (double)(hz * hz);

    double *data, *temp, *next;
    data = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));
    next = (double *)malloc(sizeof(double) * (nx + 2) * (ny + 2) * (nz + 2));

    long long max_dim = std::max(nx, std::max(ny, nz));
    double *buff;
    buff = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));

    double *send_buff1;
    send_buff1 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *send_buff2;
    send_buff2 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *send_buff3;
    send_buff3 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *send_buff4;
    send_buff4 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *send_buff5;
    send_buff5 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *send_buff6;
    send_buff6 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));

    double *receive_buff1;
    receive_buff1 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *receive_buff2;
    receive_buff2 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *receive_buff3;
    receive_buff3 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *receive_buff4;
    receive_buff4 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *receive_buff5;
    receive_buff5 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));
    double *receive_buff6;
    receive_buff6 = (double *)malloc(sizeof(double) * (max_dim + 2) * (max_dim + 2));

    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            for (int k = 0; k < nz; ++k)
            {
                data[_i(i, j, k)] = u_0;
            }
        }
    }

    // Метода Якоби
    double error = 0.0;
    do
    {
        // Вправо
        if (ib + 1 < nbx)
        {
            MPI_Request request1, request2;
            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    send_buff1[j + k * ny] = data[_i(nx - 1, j, k)];
                }
            }
            MPI_CALL(MPI_Isend(send_buff1, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD, &request1));
            MPI_CALL(MPI_Irecv(receive_buff1, ny * nz, MPI_DOUBLE, _ib(ib + 1, jb, kb), 0, MPI_COMM_WORLD, &request2));

            MPI_CALL(MPI_Wait(&request1, MPI_STATUS_IGNORE));
            MPI_CALL(MPI_Wait(&request2, MPI_STATUS_IGNORE));

            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    data[_i(nx, j, k)] = receive_buff1[j + k * ny];
                }
            }
        }
        else
        {
            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    data[_i(nx, j, k)] = u_right;
                }
            }
        }

        // Вверх
        if (jb + 1 < nby)
        {
            MPI_Request request1, request2;
            for (int k = 0; k < nz; ++k)
            {
                for (int i = 0; i < nx; ++i)
                {
                    send_buff2[i + k * nx] = data[_i(i, ny - 1, k)];
                }
            }

            MPI_CALL(MPI_Isend(send_buff2, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD, &request1));
            MPI_CALL(MPI_Irecv(receive_buff2, nx * nz, MPI_DOUBLE, _ib(ib, jb + 1, kb), 0, MPI_COMM_WORLD, &request2));

            MPI_CALL(MPI_Wait(&request1, MPI_STATUS_IGNORE));
            MPI_CALL(MPI_Wait(&request2, MPI_STATUS_IGNORE));

            for (int k = 0; k < nz; ++k)
            {
                for (int i = 0; i < nx; ++i)
                {
                    data[_i(i, ny, k)] = receive_buff2[i + k * nx];
                }
            }
        }
        else
        {
            for (int k = 0; k < nz; ++k)
            {
                for (int i = 0; i < nx; ++i)
                {
                    data[_i(i, ny, k)] = u_back;
                }
            }
        }

        // Назад
        if (kb + 1 < nbz)
        {
            MPI_Request request1, request2;
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    send_buff3[i + j * nx] = data[_i(i, j, nz - 1)];
                }
            }

            MPI_CALL(MPI_Isend(send_buff3, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD, &request1));
            MPI_CALL(MPI_Irecv(receive_buff3, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb + 1), 0, MPI_COMM_WORLD, &request2));

            MPI_CALL(MPI_Wait(&request1, MPI_STATUS_IGNORE));
            MPI_CALL(MPI_Wait(&request2, MPI_STATUS_IGNORE));

            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    data[_i(i, j, nz)] = receive_buff3[i + j * nx];
                }
            }
        }
        else
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    data[_i(i, j, nz)] = u_top;
                }
            }
        }

        // Влево
        if (ib > 0)
        {
            MPI_Request request1, request2;
            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    send_buff4[j + k * ny] = data[_i(0, j, k)];
                }
            }

            MPI_CALL(MPI_Isend(send_buff4, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD, &request1));
            MPI_CALL(MPI_Irecv(receive_buff4, ny * nz, MPI_DOUBLE, _ib(ib - 1, jb, kb), 0, MPI_COMM_WORLD, &request2));

            MPI_CALL(MPI_Wait(&request1, MPI_STATUS_IGNORE));
            MPI_CALL(MPI_Wait(&request2, MPI_STATUS_IGNORE));

            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    data[_i(-1, j, k)] = receive_buff4[j + k * ny];
                }
            }
        }
        else
        {
            for (int k = 0; k < nz; ++k)
            {
                for (int j = 0; j < ny; ++j)
                {
                    data[_i(-1, j, k)] = u_left;
                }
            }
        }

        // Вниз
        if (jb > 0)
        {
            MPI_Request request1, request2;
            for (int k = 0; k < nz; ++k)
            {
                for (int i = 0; i < nx; ++i)
                {
                    send_buff5[i + k * nx] = data[_i(i, 0, k)];
                }
            }

            MPI_CALL(MPI_Isend(send_buff5, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD, &request1));
            MPI_CALL(MPI_Irecv(receive_buff5, nx * nz, MPI_DOUBLE, _ib(ib, jb - 1, kb), 0, MPI_COMM_WORLD, &request2));

            MPI_CALL(MPI_Wait(&request1, MPI_STATUS_IGNORE));
            MPI_CALL(MPI_Wait(&request2, MPI_STATUS_IGNORE));

            for (int k = 0; k < nz; ++k)
            {
                for (int i = 0; i < nx; ++i)
                {
                    data[_i(i, -1, k)] = receive_buff5[i + k * nx];
                }
            }
        }
        else
        {
            for (int k = 0; k < nz; ++k)
            {
                for (int i = 0; i < nx; ++i)
                {
                    data[_i(i, -1, k)] = u_front;
                }
            }
        }

        // Вперед
        if (kb > 0)
        {
            MPI_Request request1, request2;
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    send_buff6[i + j * nx] = data[_i(i, j, 0)];
                }
            }

            MPI_CALL(MPI_Isend(send_buff6, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD, &request1));
            MPI_CALL(MPI_Irecv(receive_buff6, nx * ny, MPI_DOUBLE, _ib(ib, jb, kb - 1), 0, MPI_COMM_WORLD, &request2));

            MPI_CALL(MPI_Wait(&request1, MPI_STATUS_IGNORE));
            MPI_CALL(MPI_Wait(&request2, MPI_STATUS_IGNORE));

            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    data[_i(i, j, -1)] = receive_buff6[i + j * nx];
                }
            }
        }
        else
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    data[_i(i, j, -1)] = u_bottom;
                }
            }
        }

        // Перевычисление
        error = 0.0;
        for (int i = 0; i < nx; ++i)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int k = 0; k < nz; ++k)
                {
                    next[_i(i, j, k)] = ((data[_i(i - 1, j, k)] + data[_i(i + 1, j, k)]) * h2x +
                                         (data[_i(i, j - 1, k)] + data[_i(i, j + 1, k)]) * h2y +
                                         (data[_i(i, j, k - 1)] + data[_i(i, j, k + 1)]) * h2z) /
                                        (2 * (h2x + h2y + h2z));
                    error = std::max(error, std::fabs(next[_i(i, j, k)] - data[_i(i, j, k)]));
                }
            }
        }

        MPI_CALL(MPI_Allreduce(&error, &error, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));

        // Обмен предыдущего и текущего слоев
        temp = next;
        next = data;
        data = temp;

    } while (error > eps);

    // Отправляем все данные процессу с ROOT_RANK
    if (id != ROOT_RANK)
    {
        for (int k = 0; k < nz; ++k)
        {
            for (int j = 0; j < ny; ++j)
            {
                for (int i = 0; i < nx; ++i)
                {
                    buff[i] = data[_i(i, j, k)];
                }
                MPI_CALL(MPI_Send(buff, nx, MPI_DOUBLE, ROOT_RANK, 0, MPI_COMM_WORLD));
            }
        }
    }
    // Вывод данных
    else
    {
        stop = std::chrono::high_resolution_clock::now();
		t += std::chrono::duration_cast<duration>(stop - start).count();
		std::cout << "time = " << t << " ms" << std::endl;
        
        FILE *file;
        file = fopen(filename, "w");

        for (kb = 0; kb < nbz; ++kb)
        {
            for (int k = 0; k < nz; ++k)
            {
                for (jb = 0; jb < nby; ++jb)
                {
                    for (int j = 0; j < ny; ++j)
                    {
                        for (ib = 0; ib < nbx; ++ib)
                        {
                            if (_ib(ib, jb, kb) == ROOT_RANK)
                            {
                                for (int i = 0; i < nx; ++i)
                                {
                                    buff[i] = data[_i(i, j, k)];
                                }
                            }
                            else
                            {
                                MPI_CALL(MPI_Recv(buff, nx, MPI_DOUBLE, _ib(ib, jb, kb), 0, MPI_COMM_WORLD, &status));
                            }

                            for (int i = 0; i < nx; ++i)
                            {
                                fprintf(file, "%.7e ", buff[i]);
                            }
                        }
                        fprintf(file, "\n");
                    }
                }
                fprintf(file, "\n");
            }
        }
        fclose(file);
    }

    MPI_CALL(MPI_Finalize());

    free(buff);
    free(send_buff1);
    free(send_buff2);
    free(send_buff3);
    free(send_buff4);
    free(send_buff5);
    free(send_buff6);
    free(receive_buff1);
    free(receive_buff2);
    free(receive_buff3);
    free(receive_buff4);
    free(receive_buff5);
    free(receive_buff6);
    free(data);
    free(next);
    return 0;
}