/* Boost bigint_gmp.hpp header file
 *
 * Copyright 2007 Arseny Kapoulkine
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_BIGINT_BIGINT_GMP_HPP
#define BOOST_BIGINT_BIGINT_GMP_HPP

#include <limits>

#include <boost/scoped_array.hpp>

#include <boost/bigint/bigint_util.hpp>

#include <gmp.h>

namespace boost { namespace detail {
	// GMP-based implementation
	struct bigint_gmp_implementation
	{
		mpz_t data;

		bigint_gmp_implementation()
		{
			mpz_init(data);
		}

		~bigint_gmp_implementation()
		{
			mpz_clear(data);
		}

		bigint_gmp_implementation(const bigint_gmp_implementation& other)
		{
			mpz_init_set(data, other.data);
		}

		bigint_gmp_implementation& operator=(const bigint_gmp_implementation& other)
		{
			if (this != &other)
			{
				mpz_clear(data);
				mpz_init_set(data, other.data);
			}
			return *this;
		}

		void assign(int number)
		{
			mpz_set_si(data, number);
		}

		void assign(unsigned int number)
		{
			mpz_set_ui(data, number);
		}

		void assign(int64_t number)
		{
			// number is [-2^32, 2^32-1]
			// if number == -2^32, it's bit representation is 10...0, -number is 01...1+1 = 10...0 (the same)
			// converting to uint64_t yields still 10...0, it's exactly 2^32. In other cases we're safe.
			assign(static_cast<uint64_t>(number >= 0 ? number : -number));
			
			if (number < 0) mpz_neg(data, data);
		}

		void assign(uint64_t number)
		{
			mp_size_t size;
			
			data->_mp_d[0] = static_cast<mp_limb_t>(number & GMP_NUMB_MAX);
			size = number != 0;
			
			if (number > GMP_NUMB_MAX)
			{
				mpz_realloc(data, 64 / GMP_NUMB_BITS); // we know that GMP_NUMB_BITS is 2^n
				
				number >>= GMP_NUMB_BITS;
				
				while (number > 0)
				{
					data->_mp_d[size++] = static_cast<mp_limb_t>(number & GMP_NUMB_MAX);
					number >>= GMP_NUMB_BITS;
				}
			}
			
			data->_mp_size = size;
		}
		
		template <typename Ch> void _assign_str(const Ch* str, int base)
		{
			if (base < 2 && base > 36) return assign(0);
			
			// skip whitespace
			while (detail::bigint::isspace(*str)) ++str;

			int sign = 1;

			if (*str == Ch('-'))
			{
				sign = -1;
				++str;
			}
			else if (*str == Ch('+'))
			{
				++str;
			}

			static const unsigned char digit_value_tab[] =
			{
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
				0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
				0,    1,    2,    3,    4,    5,    6,    7,    8,    9,    0xff, 0xff, 0xff, 0xff, 0xff, 0xff,
				0xff, 10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,
				25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   0xff, 0xff, 0xff, 0xff, 0xff,
				0xff, 10,   11,   12,   13,   14,   15,   16,   17,   18,   19,   20,   21,   22,   23,   24,
				25,   26,   27,   28,   29,   30,   31,   32,   33,   34,   35,   0xff, 0xff, 0xff, 0xff, 0xff,
			};
			
			// skip zeros
			while (*str == Ch('0')) ++str;
			
			// is there anything left?
			if (!*str)
			{
				assign(0);
				return;
			}
			
			size_t d_size = detail::bigint::length(str);
			scoped_array<unsigned char> d(new unsigned char[d_size]);
			
			for (size_t i = 0; i < d_size; ++i)
			{
				if (!detail::bigint::is_ascii(*str) || digit_value_tab[static_cast<unsigned int>(*str)] >= base
				)
				{
					d_size = i;
					break;
				}
				
				d[i] = digit_value_tab[static_cast<unsigned int>(*str++)];
			}
			
			if (d_size == 0)
			{
				assign(0);
				return;
			}
			
			size_t d_bits = detail::bigint::get_bit_count(d_size, base);
			
			mpz_init2(data, static_cast<unsigned long>(d_bits));
			data->_mp_size = sign * mpn_set_str(data->_mp_d, d.get(), d_size, base);
		}

		void assign(const char* str, int base)
		{
			_assign_str(str, base);
		}
		
		void assign(const wchar_t* str, int base)
		{
			_assign_str(str, base);
		}

		void add(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs)
		{
			mpz_add(data, lhs.data, rhs.data);
		}

		void sub(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs)
		{
			mpz_sub(data, lhs.data, rhs.data);
		}

		void mul(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs)
		{
			mpz_mul(data, lhs.data, rhs.data);
		}

		void div(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs)
		{
			mpz_tdiv_q(data, lhs.data, rhs.data);
		}

		void mod(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs)
		{
			mpz_tdiv_r(data, lhs.data, rhs.data);
		}
		
		void or_(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs)
		{
			mpz_ior(data, lhs.data, rhs.data);
		}

		void and_(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs)
		{
			mpz_and(data, lhs.data, rhs.data);
		}

		void xor_(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs)
		{
			mpz_xor(data, lhs.data, rhs.data);
		}

		void not_(const bigint_gmp_implementation& lhs)
		{
			mpz_com(data, lhs.data);
		}

		void negate(const bigint_gmp_implementation& lhs)
		{
			mpz_neg(data, lhs.data);
		}

		void lshift(const bigint_gmp_implementation& lhs, boost::uint64_t rhs)
		{
			unsigned long max_arg = (std::numeric_limits<unsigned long>::max)();

            if (rhs <= max_arg)
            {
	            mpz_mul_2exp(data, lhs.data, static_cast<unsigned long>(rhs));
	        }
	        else
	        {
	            mpz_clear(data);
	            mpz_init_set(data, lhs.data);

    	        while (rhs > 0)
	            {
					unsigned long value = (rhs > max_arg) ? max_arg : static_cast<unsigned long>(rhs);
	        	    mpz_mul_2exp(data, data, value);
	            	rhs -= value;
				}
			}
		}

		void rshift(const bigint_gmp_implementation& lhs, boost::uint64_t rhs)
		{
			unsigned long max_arg = (std::numeric_limits<unsigned long>::max)();

            if (rhs <= max_arg)
            {
	            mpz_div_2exp(data, lhs.data, static_cast<unsigned long>(rhs));
	        }
	        else
	        {
	            mpz_clear(data);
	            mpz_init_set(data, lhs.data);

    	        while (rhs > 0)
	            {	
					unsigned long value = (rhs > max_arg) ? max_arg : static_cast<unsigned long>(rhs);
	        	    mpz_div_2exp(data, data, value);
	            	rhs -= value;
				}
			}
		}

		void inc()
		{
			mpz_add_ui(data, data, 1);
		}

		void dec()
		{
			mpz_sub_ui(data, data, 1);
		}
		
		int compare(const bigint_gmp_implementation& rhs) const
		{
			return mpz_cmp(data, rhs.data);
		}

		std::string str(int base) const
		{
			if (base < 2 || base > 36) return std::string(1, '0');
			
			size_t s_size = mpz_sizeinbase(data, base) + (mpz_sgn(data) < 0);
			scoped_array<char> s(new char[s_size + 1]);
			mpz_get_str(s.get(), base, data);
			
			std::string result(s.get()); // NRVO
			return result;
		}
		
		std::wstring wstr(int base) const
		{
			if (base < 2 || base > 36) return std::wstring(1, wchar_t('0'));
			
			size_t s_size = mpz_sizeinbase(data, base) + (mpz_sgn(data) < 0);
			scoped_array<char> s(new char[s_size + 1]);
			mpz_get_str(s.get(), base, data);
			
			std::wstring result(s.get(), s.get() + strlen(s.get()));
			return result;
		}
		
		boost::uint64_t _to_uint64() const
		{
			boost::uint64_t value = 0;
			boost::uint64_t power = 1;

			int count = data->_mp_size >= 0 ? data->_mp_size : -data->_mp_size; // abs() does not work on MSVC8

			for (int i = 0; i < count; ++i)
			{
				value += data->_mp_d[i] * power;
				power <<= GMP_NUMB_BITS;
			}
			
			return value;
		}

		template <typename T> bool can_convert_to() const
		{
			// Only integer types supported
			if (!std::numeric_limits<T>::is_integer) return false;
			
			boost::uint64_t max_value;
			int count;
			
			if (mpz_sgn(data) < 0)
			{
				count = -data->_mp_size;
				max_value = static_cast<boost::uint64_t>(-static_cast<boost::int64_t>((std::numeric_limits<T>::min)()));
			}
			else
			{
				count = data->_mp_size;
				max_value = (std::numeric_limits<T>::max)();
			}

			if (static_cast<size_t>(count) * GMP_NUMB_BITS > sizeof(boost::uint64_t) * 8) // we can't fit in uint64 => we won't fit in anything else
				return false;

			return max_value >= _to_uint64();
		}
		
		template <typename T> T to_number() const
		{
			if (!std::numeric_limits<T>::is_integer) return T();
			
			boost::uint64_t value = _to_uint64();
						
			return data->_mp_size >= 0 ? static_cast<T>(value) : static_cast<T>(-static_cast<boost::int64_t>(value));
		}

		bool is_zero() const
		{
			return mpz_sgn(data) == 0;
		}
		
		void abs(const bigint_gmp_implementation& rhs)
		{
			mpz_abs(data, rhs.data);
		}
		
		void pow(const bigint_gmp_implementation& lhs, boost::uint64_t rhs)
		{
			unsigned long max_arg = (std::numeric_limits<unsigned long>::max)();

            if (rhs <= max_arg)
            {
	            mpz_pow_ui(data, lhs.data, static_cast<unsigned long>(rhs));
	        }
	        else
	        {
	            mpz_clear(data);
	            mpz_init_set_ui(data, 1);

				mpz_t temp;

				mpz_init(temp);

    	        while (rhs > 0)
	            {	
					unsigned long value = (rhs > max_arg) ? max_arg : static_cast<unsigned long>(rhs);
	        	    mpz_pow_ui(temp, lhs.data, value);
	        	    mpz_mul(data, data, temp);
	            	rhs -= value;
				}

				mpz_clear(temp);
			}
		}
		
		void div(const bigint_gmp_implementation& lhs, const bigint_gmp_implementation& rhs, bigint_gmp_implementation& remainder)
		{
			mpz_tdiv_qr(data, remainder.data, lhs.data, rhs.data);
		}
		
		void sqrt(const bigint_gmp_implementation& lhs)
		{
			mpz_sqrt(data, lhs.data);
		}
	};
} }  // namespace boost::detail

#endif // BOOST_BIGINT_BIGINT_GMP_HPP
