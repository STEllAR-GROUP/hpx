/* Boost bigint_fft_multiplicator.hpp header file
 *
 * Copyright 2007 Arseny Kapoulkine
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_BIGINT_BIGINT_FFT_MULTIPLICATOR_HPP
#define BOOST_BIGINT_BIGINT_FFT_MULTIPLICATOR_HPP

#include <string.h>

#include <exception>
#include <algorithm>

#include <boost/cstdint.hpp>
#include <boost/bigint/bigint_default_config.hpp>

namespace boost { namespace detail {

#include "fft_primes.hpp"

static const size_t fft_iterative_threshold = 512 * 1024 / sizeof(uint32_t); // assuming 512 Kb L2 cache usage

template <size_t limb_bit_number = 32> struct bigint_fft_multiplicator
{
	typedef typename bigint_default_implementation_config<limb_bit_number>::first limb_t;
	typedef typename bigint_default_implementation_config<limb_bit_number>::second limb2_t;

	static uint32_t modmul(uint32_t a, uint32_t b, uint32_t prime, double inv_prime)
	{
	#ifdef BOOST_BIGINT_HAS_NATIVE_INT64
		return static_cast<uint32_t>(static_cast<uint64_t>(a) * b % prime);
	#else
		int r = a * b - prime * static_cast<uint32_t>(inv_prime * static_cast<int>(a) * b);
		r = (r < 0 ? r + prime : r > static_cast<int>(prime) ? r - prime : r);
		BOOST_ASSERT(static_cast<uint32_t>(r) == static_cast<uint32_t>(static_cast<uint64_t>(a) * b % prime));
		return r;
	#endif
}

	static uint32_t modadd(uint32_t a, uint32_t b, uint32_t prime)
	{
		uint32_t r = a + b;
		return (r >= prime ? r - prime : r);
	}

	static uint32_t modsub(uint32_t a, uint32_t b, uint32_t prime)
	{
		int r = static_cast<int>(a) - static_cast<int>(b);
		return (r < 0 ? r + prime : r);
	}
	
	static size_t log2(size_t value)
	{
		size_t r = 0;

		while (value > 1)
		{
			++r;
			value >>= 1;
		}

		return r;
	}

	// reverse(x) -> reverse(x+1)
	static size_t rev_next(size_t x, size_t n)
	{
		do
		{
			n >>= 1;
			x ^= n;
		}
		while ((x & n) == 0);

		return x;
	}

	// swaps values with indices that have reversed bit representation (data[1100b] <-> data[0011b])
	template <typename T> static void fft_reorder(T* data, size_t size)
	{
		if (size <= 2) return;

		size_t r = 0;

		for (size_t x = 1; x < size; ++x)
		{
			r = rev_next(r, size);

			if (r > x) std::swap(data[x], data[r]);
		}
	}

	static void fft_iterative(uint32_t* data, size_t size, uint32_t prime, double inv_prime, const uint32_t* roots)
	{
		size_t step = 1;

		uint32_t nstep = 0;

		while (step < size)
		{
			uint32_t root = roots[++nstep];

			size_t half_step = step;
			step *= 2;

			uint32_t r = 1;

			for (size_t j = 0; j < half_step; ++j)
			{
				for (size_t i = j; i < size; i += step)
				{
					uint32_t a = data[i];
					uint32_t b = modmul(data[i + half_step], r, prime, inv_prime);

					data[i] = modadd(a, b, prime);
					data[i + half_step] = modsub(a, b, prime);
				}
				
				r = modmul(r, root, prime, inv_prime);
			}
		}
	}

	static void fft_recursive(uint32_t* data, size_t size, uint32_t prime, double inv_prime, const uint32_t* roots_start, const uint32_t* roots)
	{
		if (size <= fft_iterative_threshold)
		{
			return fft_iterative(data, size, prime, inv_prime, roots_start);
		}

		size_t half_size = size / 2;

		if (half_size > 1)
		{
			fft_recursive(data, half_size, prime, inv_prime, roots_start, roots - 1);
			fft_recursive(data + half_size, half_size, prime, inv_prime, roots_start, roots - 1);
		}

		uint32_t root = *roots;
		uint32_t r = 1;

		for (size_t i = 0; i < half_size; ++i)
		{
			uint32_t a = data[i];
			uint32_t b = modmul(data[i + half_size], r, prime, inv_prime);

			data[i] = modadd(a, b, prime);
			data[i + half_size] = modsub(a, b, prime);

			r = modmul(r, root, prime, inv_prime);
		}
	}

	static void dft(uint32_t* dest, const uint16_t* source, size_t N, size_t log2N, uint32_t prime, const uint32_t* root_table)
	{
		const uint16_t* source_end = source + N;
		
		for (uint32_t* di = dest; source != source_end; ++di, ++source)
			*di = *source;
		
		fft_recursive(dest, N, prime, 1.0 / prime, root_table, root_table + log2N);
	}

	static void ift(uint32_t* data, size_t N, size_t log2N, uint32_t prime, const uint32_t* root_table, uint32_t inv_N)
	{
		double inv_prime = 1.0 / prime;
		
		fft_reorder(data, N);
		fft_recursive(data, N, prime, inv_prime, root_table, root_table + log2N);

		for (size_t i = 0; i < N; ++i)
		{
			data[i] = modmul(data[i], inv_N, prime, inv_prime);
		}
	}

	// CRT:

	// 64 x == 32 c0 mod 32 p0
	// 64 x == 32 c1 mod 32 p1

	// x == p0*t + c0
	// p0*t + c0 == c1 (mod p1)
	// t == (c1 - c0) div p0 (mod p1)

	// t == t0 (mod p1)

	// x == p0 * t0 + c0
	// t0 == (c1 - c0) div p0 (mod p1)

	static uint64_t fft_crt(uint32_t c0, uint32_t c1, double inv_prime_1)
	{
		uint32_t t0 = modmul(modsub(c1, c0, fft_primes[1]), fft_inv_p0_mod_p1, fft_primes[1], inv_prime_1);
		return static_cast<uint64_t>(fft_primes[0]) * t0 + c0;
	}

	static void fft_crt_carry(uint8_t* dest, size_t dest_size, const uint32_t* conv0, const uint32_t* conv1)
	{
		double inv_prime_1 = 1.0 / fft_primes[1];

		uint64_t carry = 0;

		uint8_t* dest_end = dest + dest_size;

		if (dest_size % 2 != 0) --dest_end;

		size_t i;
		
		for (i = 0; dest != dest_end; ++i)
		{	
			carry += fft_crt(conv0[i], conv1[i], inv_prime_1);

			*dest++ = static_cast<uint8_t>(carry & 0xff);
			*dest++ = static_cast<uint8_t>(static_cast<uint16_t>(carry & 0xff00) >> 8);
			carry >>= 16;
		}
		
		if (dest_size % 2 != 0)
		{
			carry += fft_crt(conv0[i], conv1[i], inv_prime_1);

			*dest++ = static_cast<uint8_t>(carry & 0xff);
		}
	}

	static void fft_crt_carry(uint16_t* dest, size_t dest_size, const uint32_t* conv0, const uint32_t* conv1)
	{
		double inv_prime_1 = 1.0 / fft_primes[1];

		uint64_t carry = 0;

		uint16_t* dest_end = dest + dest_size;

		for (size_t i = 0; dest != dest_end; ++i)
		{	
			carry += fft_crt(conv0[i], conv1[i], inv_prime_1);

			*dest++ = static_cast<uint16_t>(carry & 0xffff);
			carry >>= 16;
		}
	}

	static void fft_crt_carry(uint32_t* dest, size_t dest_size, const uint32_t* conv0, const uint32_t* conv1)
	{
		double inv_prime_1 = 1.0 / fft_primes[1];

		uint64_t carry = 0;

		uint32_t* dest_end = dest + dest_size;

		for (size_t i = 0; dest != dest_end; i += 2)
		{
			uint32_t l;

			carry += fft_crt(conv0[i], conv1[i], inv_prime_1);
			l = static_cast<limb_t>(carry & 0xffff);
			carry >>= 16;

			carry += fft_crt(conv0[i+1], conv1[i+1], inv_prime_1);
			*dest++ = (static_cast<limb_t>(carry & 0xffff) << 16) + l;
			carry >>= 16;
		}
	}

	static void fft_copy_source(uint16_t* dest, size_t N, const uint8_t* source, size_t source_size)
	{
		uint16_t* di = dest;

		const uint8_t* source_end = source + source_size;

		if (source_size % 2 != 0) --source_end;

		for (const uint8_t* i = source; i != source_end; i += 2)
		{
			*di++ = static_cast<uint16_t>(*i + (*(i+1) << 8));
		}

		if (source_size % 2 != 0)
			*di++ = *source_end;

		memset(di, 0, (dest + N - di) * sizeof(uint16_t));

		fft_reorder(dest, N);
	}

	static void fft_copy_source(uint16_t* dest, size_t N, const uint16_t* source, size_t source_size)
	{
		memcpy(dest, source, source_size * sizeof(uint16_t));
		memset(dest + source_size, 0, (N - source_size) * sizeof(uint16_t));
		fft_reorder(dest, N);
	}

	static void fft_copy_source(uint16_t* dest, size_t N, const uint32_t* source, size_t source_size)
	{
		uint16_t* di = dest;

		const uint32_t* source_end = source + source_size;
		
		for (const uint32_t* i = source; i != source_end; ++i)
		{
			*di++ = static_cast<uint16_t>(*i & 0xffff);
			*di++ = static_cast<uint16_t>(*i >> 16);
		}

		memset(di, 0, (dest + N - di) * sizeof(uint16_t));

		fft_reorder(dest, N);
	}

	static void mul(limb_t* dest, const limb_t* a, size_t a_size, const limb_t* b, size_t b_size)
	{
		size_t max_size = (std::max)(a_size, b_size);

		// round up to the nearest power of two
		size_t N = 1;
		while (N < max_size) N *= 2;

		// fix N depending on limb_type
		N = N * sizeof(limb_t) / sizeof(uint16_t);
		if (N == 0) N = 1;

		// destination size
		N *= 2;

		// can we perform FFT?
		if (N > fft_max_N) throw std::bad_alloc();

		size_t log2N = log2(N);

		uint32_t* workspace = new uint32_t[4*N];
		
		uint32_t* convs[] = {workspace, workspace + N};
		uint32_t* fft = workspace + 2*N;

		uint16_t* source_workspace = reinterpret_cast<uint16_t*>(workspace + 3*N);

		uint16_t* source_a = source_workspace;
		uint16_t* source_b = source_workspace + N;

		fft_copy_source(source_a, N, a, a_size);
		fft_copy_source(source_b, N, b, b_size);

		for (int p = 0; p < 2; ++p)
		{
			uint32_t prime = fft_primes[p];
			double inv_prime = 1.0 / prime;

			uint32_t* fft_a = fft;
			uint32_t* fft_b = convs[p];

			dft(fft_a, source_a, N, log2N, prime, fft_primitive_roots[p]);
			dft(fft_b, source_b, N, log2N, prime, fft_primitive_roots[p]);

			for (size_t i = 0; i < N; ++i)
			{
				fft_b[i] = modmul(fft_a[i], fft_b[i], prime, inv_prime);
			}

			ift(fft_b, N, log2N, prime, fft_inv_primitive_roots[p], fft_inv_N[p][log2N]);
		}

		fft_crt_carry(dest, a_size + b_size, convs[0], convs[1]);

		delete[] workspace;
	}

	static void sqr(limb_t* dest, const limb_t* a, size_t a_size)
	{
		// round up to the nearest power of two
		size_t N = 1;
		while (N < a_size) N *= 2;

		// fix N depending on limb_type
		N = N * sizeof(limb_t) / sizeof(uint16_t);
		if (N == 0) N = 1;

		// destination size
		N *= 2;

		// can we perform FFT?
		if (N > fft_max_N) throw std::bad_alloc();

		size_t log2N = log2(N);

		uint32_t* workspace = new uint32_t[2*N + N/2];
		
		uint32_t* convs[] = {workspace, workspace + N};

		uint16_t* source = reinterpret_cast<uint16_t*>(workspace + 2*N);

		fft_copy_source(source, N, a, a_size);

		for (int p = 0; p < 2; ++p)
		{
			uint32_t prime = fft_primes[p];
			double inv_prime = 1.0 / prime;

			uint32_t* fft = convs[p];

			dft(fft, source, N, log2N, prime, fft_primitive_roots[p]);

			for (size_t i = 0; i < N; ++i)
			{
				fft[i] = modmul(fft[i], fft[i], prime, inv_prime);
			}

			ift(fft, N, log2N, prime, fft_inv_primitive_roots[p], fft_inv_N[p][log2N]);
		}

		fft_crt_carry(dest, a_size * 2, convs[0], convs[1]);

		delete[] workspace;
	}
};

} }

#endif // BOOST_BIGINT_BIGINT_FFT_MULTIPLICATOR_HPP
