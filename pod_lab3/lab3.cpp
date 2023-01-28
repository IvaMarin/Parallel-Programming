#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>

using duration = std::chrono::microseconds;

// функция меняющая местами два элемента массива
void swap(int *array, int idx1, int idx2)
{
	int value = array[idx1];
	array[idx1] = array[idx2];
	array[idx2] = value;
}

void odd_even_sort(int arr[], int n)
{
	bool is_sorted = false;

	while (!is_sorted)
	{
		is_sorted = true;

		// сортировка пузырьком для нечетных элементов
		for (int i = 1; i <= n - 2; i = i + 2)
		{
			if (arr[i] > arr[i + 1])
			{
				swap(arr, i, i + 1);
				is_sorted = false;
			}
		}

		// сортировка пузырьком для четных элементов
		for (int i = 0; i <= n - 2; i = i + 2)
		{
			if (arr[i] > arr[i + 1])
			{
				swap(arr, i, i + 1);
				is_sorted = false;
			}
		}
	}
}

int main()
{
	// считываем бинарные данные
	int n;
	fread(&n, sizeof(int), 1, stdin);

	int *array = (int *)malloc(n * sizeof(int));
	fread(array, sizeof(int), n, stdin);

	if (n > 1)
	{
		std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
		float t = 0;
		start = std::chrono::high_resolution_clock::now();

		odd_even_sort(array, n);

		stop = std::chrono::high_resolution_clock::now();
		t += std::chrono::duration_cast<duration>(stop - start).count();
		std::cout << "time = " << t << " ms" << std::endl;
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