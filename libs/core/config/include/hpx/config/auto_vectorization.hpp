//  Copyright (c) 2022 A Kishore Kumar
//  Copyright (c) 2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config/compiler_specific.hpp>

#if defined(DOXYGEN)
/// Instructs the compiler to ignore assumed vector dependencies.
#define HPX_IVDEP
/// Instructs the compiler to attempt to force-vectorize the loop
#define HPX_VECTORIZE
/// The compiler will not propagate no-alias metadata of a variable marked with
/// restrict
#define HPX_RESTRICT
/// Instructs the compiler to unroll the loop
#define HPX_UNROLL
#else

#if defined(HPX_MSVC)
#define HPX_PRAGMA(x) __pragma(x)
#else
#define HPX_PRAGMA(x) _Pragma(#x)
#endif

// Use OpenMP backend for compilers that support OpenMP
#if (_OPENMP >= 201307) || (__INTEL_COMPILER >= 1600) ||                       \
    (defined(__clang__) && HPX_CLANG_VERSION >= 30700)
#define HPX_IVDEP
#define HPX_VECTORIZE HPX_PRAGMA(omp simd)
#define HPX_VECTOR_REDUCTION(CLAUSE) HPX_PRAGMA(omp simd reduction(CLAUSE))
#define HPX_DECLARE_SIMD _PSTL_PRAGMA(omp declare simd)

#define HPX_RESTRICT
#define HPX_UNROLL HPX_PRAGMA(omp simd)
#define HPX_UNROLL_N(N)

#define HPX_HAVE_VECTOR_REDUCTION

// Fallback to compiler-specific back-ends
#elif defined(HPX_INTEL_VERSION)
#define HPX_IVDEP HPX_PRAGMA(ivdep)
#define HPX_VECTORIZE HPX_PRAGMA(vector always dynamic_align novecremainder)
#define HPX_VECTOR_REDUCTION(CLAUSE)
#define HPX_DECLARE_SIMD

#define HPX_RESTRICT __restrict
#define HPX_UNROLL HPX_PRAGMA(unroll)
#define HPX_UNROLL_N(N) HPX_PRAGMA(unroll(N))

#undef HPX_HAVE_VECTOR_REDUCTION

#elif defined(HPX_CLANG_VERSION)
#define HPX_IVDEP HPX_PRAGMA(clang loop vectorize(enable))
#define HPX_VECTORIZE HPX_PRAGMA(clang loop interleave(enable))
#define HPX_VECTOR_REDUCTION(CLAUSE)
#define HPX_DECLARE_SIMD

#define HPX_RESTRICT __restrict
#define HPX_UNROLL HPX_PRAGMA(clang loop unroll(enable))
#define HPX_UNROLL_N(N) HPX_PRAGMA(clang loop unroll_count(N))

#undef HPX_HAVE_VECTOR_REDUCTION

#elif defined(HPX_GCC_VERSION)
#define HPX_IVDEP HPX_PRAGMA(GCC ivdep)
#define HPX_VECTORIZE
#define HPX_VECTOR_REDUCTION(CLAUSE)
#define HPX_DECLARE_SIMD

#define HPX_RESTRICT __restrict__
// GCC does not have an auto unroll constant picker
#define HPX_UNROLL HPX_PRAGMA(GCC unroll 8)
#define HPX_UNROLL_N(N) HPX_PRAGMA(GCC unroll N)

#undef HPX_HAVE_VECTOR_REDUCTION

#elif defined(HPX_MSVC)
#define HPX_IVDEP HPX_PRAGMA(loop(ivdep))
#define HPX_VECTORIZE
#define HPX_VECTOR_REDUCTION(CLAUSE)
#define HPX_DECLARE_SIMD

#define HPX_RESTRICT
#define HPX_UNROLL
#define HPX_UNROLL_N(N)

#undef HPX_HAVE_VECTOR_REDUCTION

#else

#define HPX_IVDEP
#define HPX_VECTORIZE
#define HPX_VECTOR_REDUCTION(CLAUSE)
#define HPX_DECLARE_SIMD

#define HPX_RESTRICT
#define HPX_UNROLL
#define HPX_UNROLL_N(N)

#undef HPX_HAVE_VECTOR_REDUCTION
#endif

#if defined(__AVX512F__)
#define HPX_LANE_SIZE 64
#elif defined(__AVX2__)
#define HPX_LANE_SIZE 32
#elif defined(__SSE3__)
#define HPX_LANE_SIZE 16
#else
#define HPX_LANE_SIZE 64
#endif

#endif
