/* Copyright 2020 Stanford University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <cassert>
#include <cmath>
#include <cstring>

#if (__AVX2__ == 1) || (__AVX__ == 1)
#include <immintrin.h>
#endif

#include "core.h"
#include "core_kernel.h"
#include "core_random.h"

#ifdef USE_BLAS_KERNEL
#include <mkl.h>
#endif

void execute_kernel_empty(Kernel const& kernel) {}

long long execute_kernel_busy_wait(Kernel const& kernel)
{
    long long acc = 113;
    for (long iter = 0; iter < kernel.iterations; iter++)
    {
        acc = acc * 139 % 2147483647;
    }
    return acc;
}

void copy(char* scratch_ptr, size_t scratch_bytes)
{
    assert(scratch_bytes % 2 == 0);

    char* base_ptr = reinterpret_cast<char*>(scratch_ptr);
    intptr_t base_ptr_addr = reinterpret_cast<intptr_t>(base_ptr);
    size_t prolog_padding = 0;
    char* aligned_ptr = base_ptr;
    char* prolog_ptr = base_ptr;
    char* epilog_ptr = base_ptr;
    int i;

    assert(base_ptr_addr % 2 == 0);

    if (base_ptr_addr % 32 != 0)
    {
        prolog_padding = 32 - (base_ptr_addr % 32);
        aligned_ptr = base_ptr + prolog_padding;
    }

    size_t nbytes_aligned = (scratch_bytes - prolog_padding) / 64 * 64;

    size_t epilog_padding = scratch_bytes - prolog_padding - nbytes_aligned;
    epilog_ptr = aligned_ptr + nbytes_aligned;

    char* prolog_src = prolog_ptr;
    char* prolog_dst = prolog_ptr + prolog_padding / 2;
    size_t prolog_cp_size = prolog_padding / 2;

    char* epilog_src = epilog_ptr;
    char* epilog_dst = epilog_ptr + epilog_padding / 2;
    size_t epilog_cp_size = epilog_padding / 2;

    assert(prolog_padding + nbytes_aligned + epilog_padding == scratch_bytes);

    if (prolog_padding != 0)
    {
        memcpy(prolog_dst, prolog_src, prolog_cp_size);
    }
    char* aligned_src = aligned_ptr;
    char* aligned_dst = aligned_ptr + nbytes_aligned / 2;
    assert((intptr_t) aligned_src % 32 == 0);
    assert((intptr_t) aligned_dst % 32 == 0);
#if (__AVX2__ == 1) || (__AVX__ == 1)
    size_t nb_m256 = nbytes_aligned / 64;
    for (i = 0; i < nb_m256; i++)
    {
        __m256d* dst_m256 = reinterpret_cast<__m256d*>(aligned_dst);
        double temp[4];
        std::memcpy(temp, aligned_src, sizeof(temp));
        *dst_m256 = _mm256_load_pd(temp);
        aligned_src += 32;
        aligned_dst += 32;
    }
#else
    memcpy(aligned_dst, aligned_src, nbytes_aligned / 2);
#endif
    if (epilog_padding != 0)
    {
        memcpy(epilog_dst, epilog_src, epilog_cp_size);
    }
}

void execute_kernel_memory(Kernel const& kernel, char* scratch_ptr,
    size_t scratch_bytes, long timestep)
{
#if 1
    long iter = 0;

    size_t sample_bytes = scratch_bytes / kernel.samples;

    {
        long start_idx = (timestep * kernel.iterations + iter) % kernel.samples;
        long stop_idx =
            std::min((long) kernel.samples, start_idx + kernel.iterations);
        long num_iter = stop_idx - start_idx;

        if (num_iter > 0)
        {
            char* sample_ptr = scratch_ptr + start_idx * sample_bytes;

            copy(sample_ptr, num_iter * sample_bytes);

            iter += num_iter;
        }
    }

    for (; iter + kernel.samples <= kernel.iterations; iter += kernel.samples)
    {
        long start_idx = (timestep * kernel.iterations + iter) % kernel.samples;
        long num_iter = kernel.samples;

        char* sample_ptr = scratch_ptr + start_idx * sample_bytes;

        copy(sample_ptr, num_iter * sample_bytes);
    }

    {
        long start_idx = (timestep * kernel.iterations + iter) % kernel.samples;
        long stop_idx = start_idx + (kernel.iterations - iter);
        long num_iter = stop_idx - start_idx;

        if (num_iter > 0)
        {
            char* sample_ptr = scratch_ptr + start_idx * sample_bytes;

            copy(sample_ptr, num_iter * sample_bytes);

            iter += num_iter;
        }
    }

    assert(iter == kernel.iterations);
#else
    long long N = scratch_bytes / 2;
    char* src = reinterpret_cast<char*>(scratch_ptr);
    char* dst = reinterpret_cast<char*>(scratch_ptr + N);
    for (long iter = 0; iter < kernel.iterations; iter++)
    {
        memcpy(dst, src, N);
    }
#endif
}

void execute_kernel_dgemm(
    Kernel const& kernel, char* scratch_ptr, size_t scratch_bytes)
{
#ifdef USE_BLAS_KERNEL
    long long N = scratch_bytes / (3 * sizeof(double));
    int m, n, p;
    double alpha, beta;

    m = n = p = sqrt(N);
    alpha = 1.0;
    beta = 1.0;

    assert(reinterpret_cast<uintptr_t>(scratch_ptr) % alignof(double) == 0);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    double* A = static_cast<double*>(static_cast<void*>(scratch_ptr));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    double* B = static_cast<double*>(static_cast<void*>(scratch_ptr + N * sizeof(double)));
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    double* C = static_cast<double*>(static_cast<void*>(scratch_ptr + 2 * N * sizeof(double)));

    for (long iter = 0; iter < kernel.iterations; iter++)
    {
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, p, alpha,
            A, p, B, n, beta, C, n);
    }
#else
    fprintf(stderr, "No BLAS is detected\n");
    fflush(stderr);
    abort();
#endif
}

void execute_kernel_daxpy(Kernel const& kernel, char* scratch_large_ptr,
    size_t scratch_large_bytes, long timestep)
{
#ifdef USE_BLAS_KERNEL
    for (long iter = 0; iter < kernel.iterations; iter++)
    {
        size_t scratch_bytes = scratch_large_bytes / kernel.samples;
        int idx = (timestep * kernel.iterations + iter) % kernel.samples;
        char* scratch_ptr = scratch_large_ptr + idx * scratch_bytes;

        int N = scratch_bytes / (2 * sizeof(double));
        double alpha;

        alpha = 1.0;

        assert(reinterpret_cast<uintptr_t>(scratch_ptr) % alignof(double) == 0);
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        double* X = static_cast<double*>(static_cast<void*>(scratch_ptr));
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
        double* Y = static_cast<double*>(static_cast<void*>(scratch_ptr + N * sizeof(double)));

        cblas_daxpy(N, alpha, X, 1, Y, 1);
    }
#else
    fprintf(stderr, "No BLAS is detected\n");
    fflush(stderr);
    abort();
#endif
}

double execute_kernel_compute(Kernel const& kernel)
{
#if __AVX2__ == 1
    __m256d A[16];

    for (int i = 0; i < 16; i++)
    {
        A[i] = _mm256_set_pd(1.0f, 2.0f, 3.0f, 4.0f);
    }

    for (long iter = 0; iter < kernel.iterations; iter++)
    {
        for (int i = 0; i < 16; i++)
        {
            A[i] = _mm256_fmadd_pd(A[i], A[i], A[i]);
        }
    }
#elif __AVX__ == 1
    __m256d A[16];

    for (int i = 0; i < 16; i++)
    {
        A[i] = _mm256_set_pd(1.0f, 2.0f, 3.0f, 4.0f);
    }

    for (long iter = 0; iter < kernel.iterations; iter++)
    {
        for (int i = 0; i < 16; i++)
        {
            A[i] = _mm256_mul_pd(A[i], A[i]);
            A[i] = _mm256_add_pd(A[i], A[i]);
        }
    }
#else
    double A[64];

    for (int i = 0; i < 64; i++)
    {
        A[i] = 1.2345;
    }

    for (long iter = 0; iter < kernel.iterations; iter++)
    {
        for (int i = 0; i < 64; i++)
        {
            A[i] = A[i] * A[i] + A[i];
        }
    }
#endif
    double* C = static_cast<double*>(A);
    double dot = 1.0;
    for (int i = 0; i < 64; i++)
    {
        dot *= C[i];
    }
    return dot;
}

double execute_kernel_compute2(Kernel const& kernel)
{
    constexpr size_t N = 32;
    double A[N] = {0};
    double B[N] = {0};
    double C[N] = {0};

    for (size_t i = 0; i < N; ++i)
    {
        A[i] = 1.2345;
        B[i] = 1.010101;
    }

    for (long iter = 0; iter < kernel.iterations; iter++)
    {
        for (size_t i = 0; i < N; ++i)
        {
            C[i] = C[i] + (A[i] * B[i]);
        }
    }

    double sum = 0;
    for (size_t i = 0; i < N; ++i)
    {
        sum += C[i];
    }
    return sum;
}

void execute_kernel_io(Kernel const& kernel)
{
    assert(false);
}

long select_imbalance_iterations(
    Kernel const& kernel, long graph_index, long timestep, long point)
{
    long seed[3] = {graph_index, timestep, point};
    double value = random_uniform(&seed[0], sizeof(seed));

    long iterations = (long) round(
        (1 + (value - 0.5) * kernel.imbalance) * kernel.iterations);
    assert(iterations >= 0);
    return iterations;
}

double execute_kernel_imbalance(
    Kernel const& kernel, long graph_index, long timestep, long point)
{
    long iterations =
        select_imbalance_iterations(kernel, graph_index, timestep, point);
    Kernel k(kernel);
    k.iterations = iterations;

    return execute_kernel_compute(k);
}
