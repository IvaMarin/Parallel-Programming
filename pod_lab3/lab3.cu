#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

// обработчик ошибок cudaError_t
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

// введение фиктивных элементов для разрешения конфликтов банков памяти
#define _index(i) ((i) + ((i) >> 5))

const int BLOCKS = 256;
const int BLOCK_SIZE = 1024;

// функция для заполнения массива значениями INT_MAX до размера кратного BLOCK_SIZE
__global__ void fill_array(int *array, int n, int value)
{
	// вычисляем абсолютный номер потока
	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	// вычисляем число потоков - это будет наш шаг, если потоков меньше чем n
	int offset = blockDim.x * gridDim.x;

	for (int i = idx; i < n; i += offset)
	{
		array[i] = value;
	}
}

// функция меняющая местами два элемента массива
__device__ void swap(int *array, int idx1, int idx2)
{
	int value = array[idx1];
	array[idx1] = array[idx2];
	array[idx2] = value;
}

// функция реализующая полуочиститель B(n)
__device__ void half_cleaner(int *buff, int M, int B, unsigned int idx1)
{
	// ровно половина потоков будет задействована,
	// при этом для каждого полуочистителя это будут разные потоки, что уменьшит дивергенцию потоков
	unsigned int idx2 = idx1 ^ B;
	if (idx2 > idx1)
	{
		if ((idx1 & M) == 0) // сортируем по возрастанию
		{
			if (buff[_index(idx1)] > buff[_index(idx2)])
				swap(buff, _index(idx1), _index(idx2));
		}
		else // сортируем по убыванию
		{
			if (buff[_index(idx1)] < buff[_index(idx2)])
				swap(buff, _index(idx1), _index(idx2));
		}
	}
}

// функция реализующая полуочиститель B(n) для каждого блока
__global__ void bitonic_sort(int *array, int n, int M, int B)
{
	__shared__ int buff[_index(BLOCK_SIZE)];

	int *block;
	unsigned int idx = threadIdx.x;
	for (int shift = blockIdx.x * BLOCK_SIZE; shift < n; shift += gridDim.x * BLOCK_SIZE) // цикл по блокам
	{
		// копируем данные в разделяемую память
		block = array + shift;
		buff[_index(threadIdx.x)] = block[threadIdx.x];
		__syncthreads();

		// полуочиститель B(n)
		half_cleaner(buff, M, B, idx);
		__syncthreads();

		// копируем обратно блок из битонических последовательностей длины B
		block[idx] = buff[_index(idx)];
	}
}

// функция реализующая битоническое слияние M(BLOCK_SIZE) для четных-нечетных пар блоков
__global__ void even_odd_merge(int *array, int n, int start, int end)
{
	__shared__ int buff[_index(BLOCK_SIZE)];

	int *block;
	unsigned int idx = threadIdx.x;
	start += blockIdx.x * BLOCK_SIZE;
	for (int shift = start; shift < end; shift += gridDim.x * BLOCK_SIZE)
	{
		block = array + shift;

		// первую половину блока загружаем в прямом порядке
		// первая половина битоничесокой последоветельности
		if (idx < BLOCK_SIZE / 2)
		{
			buff[_index(idx)] = block[idx];
		}
		// вторую половину блока загружаем в обратном порядке
		// вторая половина битоничесокой последоветельности
		else
		{
			buff[_index(idx)] = block[(BLOCK_SIZE * 3 / 2 - 1) - idx];
		}
		__syncthreads();

		// выполняем битоническое слияние M(BLOCK_SIZE)
		for (int B = BLOCK_SIZE; B >= 2; B >>= 1) // цикл по полуочистителям B(n)
		{
			half_cleaner(buff, BLOCK_SIZE, (B >> 1), idx);
			__syncthreads();
		}

		// копируем обратно блок (отсортированную последоваетльность длины BLOCK_SIZE)
		block[idx] = buff[_index(idx)];
	}
}

int main()
{
	// считываем бинарные данные
	int n;
	fread(&n, sizeof(int), 1, stdin);

	int *array = (int *)malloc(n * sizeof(int));
	fread(array, sizeof(int), n, stdin);

	if (n > 0)
	{
		// вычисляем размер последнего неполного блока
		int last_block_size = n & ((1 << 10) - 1);
		int new_n = n;

		// обновляем размер массива до кратного BLOCK_SIZE
		if (last_block_size > 0)
		{
			new_n -= last_block_size;
			new_n += BLOCK_SIZE;
		}

		int *dev_array;
		CSC(cudaMalloc(&dev_array, sizeof(int) * new_n));

		CSC(cudaMemcpy(dev_array, array, sizeof(int) * n, cudaMemcpyHostToDevice));

		// запоняем массив значениями INT_MAX до размера кратного BLOCK_SIZE, если это необходимо
		if (last_block_size > 0)
		{
			fill_array<<<BLOCKS, BLOCK_SIZE>>>(dev_array + n, new_n - n, INT_MAX);
			CSC(cudaGetLastError());
		}

		cudaEvent_t start, stop;
		CSC(cudaEventCreate(&start));
		CSC(cudaEventCreate(&stop));
		CSC(cudaEventRecord(start));

		// Блочная сортировка чет-нечет:

		// Этап 1: Битоническая сортировка блоков
		for (int M = 2; M <= BLOCK_SIZE; M <<= 1) // цикл по битоническим слияниям M(n)
		{
			for (int B = M; B >= 2; B >>= 1) // цикл по полуочистителям B(n)
			{
				bitonic_sort<<<BLOCKS, BLOCK_SIZE>>>(dev_array, new_n, M, (B >> 1));
				CSC(cudaGetLastError());
			}
		}

		// в случае, если размер массива равен BLOCK_SIZE, то на этом сортировка завершается
		int blocks_count = new_n / BLOCK_SIZE;
		if (blocks_count > 1)
		{
			// Этап 2: Битонические  слияния
			for (int i = 0; i < blocks_count; i++)
			{
				// сливаем четные пары блоков
				even_odd_merge<<<BLOCKS, BLOCK_SIZE>>>(dev_array, new_n, 0, new_n);
				CSC(cudaGetLastError());

				// сливаем нечетные пары блоков
				even_odd_merge<<<BLOCKS, BLOCK_SIZE>>>(dev_array, new_n, (BLOCK_SIZE / 2), new_n - BLOCK_SIZE);
				CSC(cudaGetLastError());
			}
		}

		CSC(cudaEventRecord(stop));
		CSC(cudaEventSynchronize(stop));

		float t;
		CSC(cudaEventElapsedTime(&t, start, stop));
		CSC(cudaEventDestroy(start));
		CSC(cudaEventDestroy(stop));

		printf("time = %f ms\n", t);

		CSC(cudaMemcpy(array, dev_array, sizeof(int) * n, cudaMemcpyDeviceToHost));

		// Освобождаем память GPU
		CSC(cudaFree(dev_array));
	}

	// fwrite(array, sizeof(int), n, stdout);
	// for (int i = 0; i < n; i++)
	// {
	// 	printf("%d ", array[i]);
	// }
	// printf("\n");

	// Освобождаем память СPU
	free(array);
	return 0;
}