#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <chrono>

using duration = std::chrono::microseconds;

bool compare_by_absolute_value(const double &a, const double &b)
{
	return fabs(a) < fabs(b);
}

// функция меняющая местами две строки
void swap_rows(double *system, int curr_id, int max_id, int n)
{
	double element;
	for (int j = 0; j < n + 1; j++)
	{
		element = system[j * n + curr_id];
		system[j * n + curr_id] = system[j * n + max_id];
		system[j * n + max_id] = element;
	}
}

// функция "зануления" всех элементов ниже данного
void subtract_row_from_rows_below(double *system, int curr_id, int n)
{
	double coefficient;
	for (int i = curr_id + 1; i < n; i++)
	{
		coefficient = system[curr_id * n + i] / system[curr_id * n + curr_id];
		for (int j = curr_id + 1; j < n + 1; j++)
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

	int max_id;

	std::chrono::time_point<std::chrono::high_resolution_clock> start, stop;
    float t = 0;
    start = std::chrono::high_resolution_clock::now();

	// прямой ход метода Гаусса
	for (int i = 0; i < n - 1; i++)
	{
		auto system_ptr = system + i * n;
		
		auto max_id_ptr = std::max_element(system_ptr + i, system_ptr + n, compare_by_absolute_value);
		max_id = max_id_ptr - system_ptr;

		if (i != max_id)
		{
			swap_rows(system, i, max_id, n);
		}
		subtract_row_from_rows_below(system, i, n);
	}

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

	stop = std::chrono::high_resolution_clock::now();
    t += std::chrono::duration_cast<duration>(stop - start).count();
    std::cout << "time = " << t << " ms" << std::endl;

	// for (int i = 0; i < n; i++)
	// {
	// 	printf("%.10e ", x[i]);
	// }
	// printf("\n");

	// Освобождаем память СPU
	free(x);
	free(system);
	return 0;
}