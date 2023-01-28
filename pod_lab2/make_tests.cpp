// g++ -std=c++11 -O2 -Wextra -Wall -Wno-sign-compare -Wno-unused-result -o make_tests make_tests.cpp

#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>

#include "matrix.hpp"

std::ofstream generated_file;

int main()
{
	std::vector<int> tests = {10, 100, 1000, 3162};
	srand(time(NULL));
	for (auto t : tests)
    {
		size_t _r = t;
		size_t c = t;

		std::string generated_file_name = "tests/test_" + std::to_string(_r*c) + ".t";
		generated_file.open(generated_file_name);
		generated_file.precision(6);
		generated_file << std::fixed;

		generated_file << c << std::endl;

		Matrix<double> A(_r, c);
		for (size_t i = 0; i < _r; i++)
		{
			for (size_t j = 0; j < c; j++)
			{
				double r = double(rand() % 10);
				A[i][j] = r;
			}
		}

		for (size_t i = 0; i < _r; i++)
		{
			double max = 0.0;
			for (size_t j = 0; j < c; j++)
			{
				if (max < A[i][j])
					max = A[i][j];
			}
			A[i][i] = max * _r + 1.0;
		}

		generated_file.precision(0);
		generated_file << std::fixed;
		generated_file << A;

		std::vector<double> b(_r, 0);
		for (size_t i = 0; i < _r; i++)
		{
			double r = double(rand());
			b[i] = r;
			generated_file << b[i] << " ";
		}
		generated_file.close();
	}
	return 0;
}