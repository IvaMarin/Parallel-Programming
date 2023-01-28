#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <iostream>
#include <fstream>
#include <vector>
#include <cassert>

template <typename Type>
class Matrix
{
public:
	Matrix() : row(0), col(0) { _matrix.resize(0); }
	Matrix(size_t row, size_t col) { init(row, col); }
	Matrix(const Matrix<Type> &A)
	{
		row = A.rows();
		col = A.cols();
		init(row, col);
		for (size_t i = 0; i < row; i++)
		{
			for (size_t j = 0; j < col; j++)
				_matrix[i][j] = A[i][j];
		}
	}

	// accessors
	inline std::vector<Type> &operator[](size_t i) { return _matrix[i]; }
	inline Type &operator()(size_t row, size_t column) { return _matrix[row][column]; }
	inline const std::vector<Type> &operator[](size_t i) const { return _matrix[i]; }
	inline const Type &operator()(size_t row, size_t column) const { return _matrix[row][column]; }

	size_t rows() const { return row; }
	size_t cols() const { return col; }

	friend std::ifstream &operator>>(std::ifstream &in, Matrix<Type> &matr)
	{
		for (size_t i = 0; i < matr.rows(); ++i)
		{
			for (size_t j = 0; j < matr.cols(); ++j)
				in >> matr[i][j];
		}
		return in;
	}

	friend std::ostream &operator<<(std::ostream &out, const Matrix<Type> &matr)
	{
		for (size_t i = 0; i < matr.rows(); ++i)
		{
			for (size_t j = 0; j < matr.cols(); ++j)
			{
				if (j)
					out << " ";
				out << matr[i][j];
			}
			out << std::endl;
		}
		return out;
	}

	friend Matrix<Type> operator+(const Matrix<Type> &A1, const Matrix<Type> &A2)
	{
		Matrix<Type> tmp(A1);
		for (size_t i = 0; i < A1.rows(); i++)
		{
			for (size_t j = 0; j < A1.cols(); j++)
				tmp[i][j] = A1[i][j] + A2[i][j];
		}
		return tmp;
	}

	friend Matrix<Type> operator-(const Matrix<Type> &A1, const Matrix<Type> &A2)
	{
		Matrix<Type> tmp(A1);
		for (size_t i = 0; i < A1.rows(); i++)
		{
			for (size_t j = 0; j < A1.cols(); j++)
				tmp[i][j] = A1[i][j] - A2[i][j];
		}
		return tmp;
	}

	friend Matrix<Type> operator*(const Matrix<Type> &A, const Type &x)
	{
		size_t row = A.rows();
		size_t col = A.cols();
		Matrix<Type> tmp(row, col);
		for (size_t i = 0; i < row; i++)
		{
			for (size_t j = 0; j < col; j++)
				tmp[i][j] = x * A[i][j];
		}

		return tmp;
	}

	friend Matrix<Type> operator*(const Type &x, const Matrix<Type> &A) { return A * x; }

	friend Matrix<Type> operator*(const Matrix<Type> &A1, const Matrix<Type> &A2)
	{
		assert(A1.cols() == A2.rows());
		size_t rows = A1.rows();
		size_t columns = A2.cols();
		size_t K = A1.cols();
		Matrix<Type> tmp(rows, columns);
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < columns; ++j)
			{
				tmp[i][j] = 0;
				for (size_t k = 0; k < K; ++k)
					tmp[i][j] += A1[i][k] * A2[k][j];
			}

		return tmp;
	}

	friend std::vector<Type> operator*(const Matrix<Type> &A, const std::vector<Type> &b)
	{
		assert(A.cols() == b.size());

		size_t rows = A.rows();
		size_t columns = A.cols();
		std::vector<Type> tmp(rows);
		for (size_t i = 0; i < rows; ++i)
		{
			Type sum = 0;
			for (size_t j = 0; j < columns; ++j)
				sum += A[i][j] * b[j];
			tmp[i] = sum;
		}

		return tmp;
	}

	friend std::vector<Type> operator*(const std::vector<Type> &b, const Matrix<Type> &A)
	{
		assert(A.rows() == b.size());

		const size_t cols = A.cols();
		const size_t size = A.rows();
		std::vector<Type> tmp(cols);
		for (size_t i = 0; i < cols; ++i)
		{
			Type sum = 0;
			for (size_t j = 0; j < size; ++j)
				sum += A[i][j] * b[j];
			tmp[i] = sum;
		}

		return tmp;
	}

	friend Matrix<Type> operator/(const Matrix<Type> &A, const Type &x)
	{
		const size_t rows = A.rows();
		const size_t clumns = A.cols();

		Matrix<Type> tmp(rows, clumns);
		for (size_t i = 0; i < rows; ++i)
			for (size_t j = 0; j < clumns; ++j)
				tmp[i][j] = A[i][j] / x;

		return tmp;
	}

	friend Matrix<Type> operator/(const Type &x, const Matrix<Type> &A)
	{
		const size_t rows = A.rows();
		const size_t clumns = A.cols();

		Matrix<Type> tmp(rows, clumns);
		for (size_t i = 0; i < rows; ++i)
		{
			for (size_t j = 0; j < clumns; ++j)
				tmp[i][j] = x / A[i][j];
		}

		return tmp;
	}

private:
	std::vector<std::vector<Type>> _matrix;
	size_t row;
	size_t col;
	void init(size_t rows, size_t columns)
	{
		row = rows;
		col = columns;
		_matrix.resize(row);
		for (size_t i = 0; i < row; i++)
		{
			_matrix[i].resize(col);
			for (size_t j = 0; j < col; j++)
				_matrix[i][j] = 0.0;
		}
	}
};

#endif /* MATRIX_HPP */