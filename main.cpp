//#define BENCHPRESS_CONFIG_MAIN
#define CATCH_CONFIG_MAIN

#include "Multilpyer.h"
#include <vector>
#include <iostream>
//#include "benchpress\benchpress.hpp"
#include "catch.hpp"

using std::vector;

//Benchpress

/*BENCHMARK("1 thread", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 1);
	}
});

BENCHMARK("1 thread, with transpose", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		A.Transpose();
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 1);
	}
});

BENCHMARK("2 thread", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 2);
	}
});

BENCHMARK("2 thread, with transpose", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		A.Transpose();
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 2);
	}
});

BENCHMARK("3 thread", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 3);
	}
});

BENCHMARK("3 thread, with transpose", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		A.Transpose();
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 3);
	}
});

BENCHMARK("4 thread", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 4);
	}
});

BENCHMARK("4 thread, with transpose", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		A.Transpose();
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 4);
	}
});

BENCHMARK("5 thread", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 5);
	}
});

BENCHMARK("5 thread, with transpose", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		A.Transpose();
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 5);
	}
});

BENCHMARK("6 thread", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 6);
	}
});

BENCHMARK("6 thread, with transpose", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		A.Transpose();
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 6);
	}
});

BENCHMARK("7 thread", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 7);
	}
});

BENCHMARK("7 thread, with transpose", [](benchpress::context* ctx) {
	for (size_t i = 0; i < ctx->num_iterations(); ++i) {
		Mul A = Mul(vector<vector<double>>(25, vector<double>(25, 4)));
		A.Transpose();
		Mul B = Mul(vector<vector<double>>(25, vector<double>(25, 18)));
		A.Multiply(B, 7);
	}
});

BENCHMARK("wait", [](benchpress::context* ctx) {
	int a;
	std::cin >> a;
});*/

//Catch

TEST_CASE("Creating") {
	vector<vector<double>> vecA = { { 1, 3 },{ 1, 2 } };
	Mul A = Mul(vecA);
	REQUIRE(A.matrix == vecA);
	REQUIRE(A.transposed == false);
}

TEST_CASE("Simple multiplication") {
  vector<vector<double>> vecA = { { 1, 3 },{ 1, 2 } };
  Mul A = Mul(vecA);
  vector<vector<double>> vecB = { { 1, 2, 1 },{ 3, 1, 0 } };
  Mul B = Mul(vecB);

  vector<vector<double>> ans = { { 10, 5, 1 },{ 7, 4, 1 } };

  SECTION("Without transpose") {
	vector<vector<double>> result = A.Multiply(B, 1);
    REQUIRE(result == ans);
  }

  SECTION("With transpose") {
	  A.Transpose();
	  vector<vector<double>> result = A.Multiply(B, 1);
	  REQUIRE(result == ans);
  }
}

TEST_CASE("Two threads") {
	vector<vector<double>> vecA(10, vector<double>(10, 10));
	vector<vector<double>> vecB(10, vector<double>(10, 100));

	Mul A = Mul(vecA);
	Mul B = Mul(vecB);
	vector<vector<double>> ans(10, vector<double>(10, 10000));

	SECTION("Without transpose") {
		vector<vector<double>> result = A.Multiply(B, 2);
		REQUIRE(result == ans);
	}

	SECTION("With transpose") {
		A.Transpose();
		vector<vector<double>> result = A.Multiply(B, 2);
		REQUIRE(result == ans);
	}
}