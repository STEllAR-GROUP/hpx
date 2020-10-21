// Copyright (c) 2013 Erik Schnetter
//
// SPDX-License-Identifier: BSL-1.0
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#if !defined(HPX_COMPUTE_DEVICE_CODE)
#include "block_matrix.hpp"
#include "matrix.hpp"

// Level 1

// axpy: y = alpha x + y
void axpy(double alpha, const vector_t& x, vector_t& y);
void axpy(double alpha, const block_vector_t& x, block_vector_t& y);

// copy: y = x
void copy(const vector_t& x, vector_t& y);
void copy(const block_vector_t& x, block_vector_t& y);

// nrm2: sqrt(x^T x)
double nrm2_process(const vector_t& x);
double nrm2(const vector_t& x);
double nrm2(const block_vector_t& x);

// scal: x = alpha x
void scal(double alpha, vector_t& x);
void scal(double alpha, block_vector_t& x);



// Level 2

// axpy: b = alpha a + b
void axpy(bool transa, double alpha, const matrix_t& a, matrix_t& b);
void axpy(bool transa, double alpha, const block_matrix_t& a,
          block_matrix_t& b);

// copy: b = a
void copy(bool transa, const matrix_t& a, matrix_t& b);
void copy(bool transa, const block_matrix_t& a, block_matrix_t& b);

// gemv: y = alpha T[a] x + beta y
void gemv(bool trans, double alpha, const matrix_t& a, const vector_t& x,
          double beta, vector_t& y);
void gemv(bool trans, double alpha,
          const block_matrix_t& a, const block_vector_t& x,
          double beta, block_vector_t& y);

// nrm2: sqrt(trace a^T a)
double nrm2_process(const matrix_t& a);
double nrm2(const matrix_t& a);
double nrm2(const block_matrix_t& a);

// scal: a = alpha a
void scal(double alpha, matrix_t& a);
void scal(double alpha, block_matrix_t& a);



// Level 3

// gemm: c = alpha T[a] T[b] + beta c
void gemm(bool transa, bool transb,
          double alpha, const matrix_t& a, const matrix_t& b,
          double beta, matrix_t& c);
void gemm(bool transa, bool transb,
          double alpha, const block_matrix_t& a, const block_matrix_t& b,
          double beta, block_matrix_t& c);
#endif
