#pragma once
#include <vector>
#include <thread>
#include <future>

using std::vector;

class Mul {
public:
	vector<vector<double>> matrix;
	bool transposed;

	Mul(const vector<vector<double>> &input);
	void Transpose();
	void RawsMultiply(int firstRaw, int lastRaw, const vector<vector<double>> &other, vector<vector<double>> &result);
	vector<vector<double>> Multiply(const Mul &other, int threadsCount);
};

Mul::Mul(const vector<vector<double>> &input) {
	matrix = input;
}

void Mul::Transpose() {
	vector<vector<double>> transposedMatrix(matrix.size(), vector<double>(matrix[0].size()));
	for (int i = 0; i < matrix.size(); i++) {
		for (int j = 0; j < matrix[0].size(); j++) {
			transposedMatrix[i][j] = matrix[j][i];
		}
	}
	matrix = transposedMatrix;
	transposed = true;
}

void Mul::RawsMultiply(int firstRaw, int lastRaw, const vector<vector<double>> &other, vector<vector<double>> &result) {
	for (int i = firstRaw; i < lastRaw; i++) {
		for (int j = 0; j < result[i].size(); j++) {
			for (int r = 0; r < matrix[0].size(); r++) {
				if (!transposed) {
					result[i][j] += matrix[i][r] * other[r][j];
				}
				else {
					result[i][j] += matrix[r][i] * other[r][j];
				}
			}
		}
	}
}

vector<vector<double>> Mul::Multiply(const Mul &other, int threadsCount) {
	vector<vector<double>> result(matrix.size(), vector<double>(other.matrix[0].size()));
	vector<std::future<void>> futures(threadsCount);
	int size_of_part = matrix.size() / threadsCount;
	for (int i = 0; i < threadsCount; i++) {
		int from = size_of_part * i;
		int to = size_of_part * (i + 1);
		to += (i == threadsCount - 1) ? matrix.size() % threadsCount : 0;
		futures[i] = std::async(&Mul::RawsMultiply, this, from, to, other.matrix, std::ref(result));
	}
	for (int i = 0; i < futures.size(); i++) {
		futures[i].get();
	}
	return result;
}