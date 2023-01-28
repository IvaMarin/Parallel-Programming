#include <stdio.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

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

struct comparator
{
	// переопределение оператора "()" для экземпляра этой структуры
	__host__ __device__ bool operator()(double a, double b)
	{
		return fabs(a) < fabs(b);
	}
};

// функция меняющая местами две строки
__global__ void swap_rows(double *system, int curr_id, int max_id, int n)
{
	// вычисляем абсолютный номер потока
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// вычисляем число потоков - это будет наш шаг, если потоков меньше чем n
	int offset = blockDim.x * gridDim.x;

	double element;
	for (int j = idx; j < n + 1; j += offset)
	{
		element = system[j * n + curr_id];
		system[j * n + curr_id] = system[j * n + max_id];
		system[j * n + max_id] = element;
	}
}

// функция "зануления" всех элементов ниже данного
__global__ void subtract_row_from_rows_below(double *system, int curr_id, int n)
{
	// вычисляем абсолютный номер потока
	int idx = blockDim.x * blockIdx.x + threadIdx.x;
	int idy = blockDim.y * blockIdx.y + threadIdx.y;

	// вычисляем число потоков - это будет наш шаг, если потоков меньше чем n
	int offsetx = blockDim.x * gridDim.x;
	int offsety = blockDim.y * gridDim.y;

	double coefficient;
	for (int i = idx + curr_id + 1; i < n; i += offsetx)
	{
		coefficient = system[curr_id * n + i] / system[curr_id * n + curr_id];
		for (int j = idy + curr_id + 1; j < n + 1; j += offsety)
		{
			system[j * n + i] -= system[j * n + curr_id] * coefficient;
		}
	}
}

int main()
{
	int n; // размерность квадратной матрицы
	scanf("%d", &n);

	double *system = (double *)malloc(sizeof(double) * (n + 1) * n);
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			scanf("%lf", &system[i + j * n]); // сохраняем матрицу по столбцам
		}
	}
	for (int j = 0; j < n; j++)
	{
		scanf("%lf", &system[n * n + j]); // дописываем вектор свободных коэффициентов
	}

	double *dev_system;
	CSC(cudaMalloc(&dev_system, sizeof(double) * (n + 1) * n));
	CSC(cudaMemcpy(dev_system, system, sizeof(double) * (n + 1) * n, cudaMemcpyHostToDevice));

	dim3 threadsperBlock(256);
	dim3 numBlocks(256);

	dim3 threadsperBlock2D(32, 32);
	dim3 numBlocks2D(256, 256);

	comparator compare_by_absolute_value;
	thrust::device_ptr<double> max_id_ptr;
	int max_id;

	cudaEvent_t start, stop;
    CSC(cudaEventCreate(&start));
    CSC(cudaEventCreate(&stop));
    CSC(cudaEventRecord(start));

	// прямой ход метода Гаусса
	for (int i = 0; i < n - 1; i++)
	{
		// выполняем приведение типов
		thrust::device_ptr<double> system_ptr(dev_system + i * n);

		// ищем максимум в массиве на GPU
		max_id_ptr = thrust::max_element(system_ptr + i, system_ptr + n, compare_by_absolute_value);
		max_id = max_id_ptr - system_ptr;

		if (i != max_id)
		{
			swap_rows<<<numBlocks, threadsperBlock>>>(dev_system, i, max_id, n);
			CSC(cudaGetLastError());
		}
		subtract_row_from_rows_below<<<numBlocks2D, threadsperBlock2D>>>(dev_system, i, n);
		CSC(cudaGetLastError());
	}

	CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));

	float t1;
	CSC(cudaEventElapsedTime(&t1, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));
	
	CSC(cudaMemcpy(system, dev_system, sizeof(double) * (n + 1) * n, cudaMemcpyDeviceToHost));

	CSC(cudaEventCreate(&start));
	CSC(cudaEventCreate(&stop));
	CSC(cudaEventRecord(start));

	// находим вектор неизветсных х
	double *x = (double *)malloc(sizeof(double) * n);
	for (int i = n - 1; i >= 0; i--)
	{
		x[i] = system[n * n + i];
		for (int j = n - 1; j > i; j--)
		{
			x[i] -= system[i + j * n] * x[j];
		}
		x[i] /= system[i * n + i];
	}

	CSC(cudaEventRecord(stop));
	CSC(cudaEventSynchronize(stop));

	float t2;
	CSC(cudaEventElapsedTime(&t2, start, stop));
	CSC(cudaEventDestroy(start));
	CSC(cudaEventDestroy(stop));

	float t = t1 + t2;
	printf("time = %f ms\n", t);

	// for (int i = 0; i < n; i++)
	// {
	// 	printf("%.10e ", x[i]);
	// }
	// printf("\n");

	// Освобождаем память GPU
	CSC(cudaFree(dev_system));

	// Освобождаем память СPU
	free(x);
	free(system);
	return 0;
}