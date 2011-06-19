/* Boost bigint_default.hpp header file
 *
 * Copyright 2007 Arseny Kapoulkine
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_BIGINT_BIGINT_DEFAULT_HPP
#define BOOST_BIGINT_BIGINT_DEFAULT_HPP

#include <limits>

#include <boost/scoped_array.hpp>

#include <boost/bigint/bigint_util.hpp>
#include <boost/bigint/bigint_default_config.hpp>

#include <boost/bigint/bigint_fft_multiplicator.hpp>

namespace boost { namespace detail {
	// Default implementation
	template <template <class> class Storage, size_t limb_bit_number = 32> struct bigint_default_implementation
	{
		typedef typename bigint_default_implementation_config<limb_bit_number>::first limb_t;
		typedef typename bigint_default_implementation_config<limb_bit_number>::second limb2_t;

		static limb_t limb_max()
		{
			return std::numeric_limits<limb_t>::max();
		}

		Storage<limb_t> data;
		bool negative;

		bigint_default_implementation(): negative(false)
		{
		}

		void assign(int number)
		{
			assign(static_cast<int64_t>(number));
		}

		void assign(unsigned int number)
		{
			assign(static_cast<uint64_t>(number));
		}

		void assign(int64_t number)
		{
			// number is [-2^32, 2^32-1]
			// if number == -2^32, it's bit representation is 10...0, -number is 01...1+1 = 10...0 (the same)
			// converting to uint64_t yields still 10...0, it's exactly 2^32. In other cases we're safe.
			assign(static_cast<uint64_t>(number >= 0 ? number : -number));
			
			negative = (number < 0);
		}

		void assign(uint64_t number)
		{
			size_t size = 0;
			
			if (number != 0)
			{
				data.resize(1);
				data[0] = static_cast<limb_t>(number & limb_max());

				size = 1;
			}

			if (number > limb_max())
			{
				data.resize(64 / limb_bit_number); // we know that limb_bit_number is 2^n
				
				number >>= limb_bit_number;
				
				while (number > 0)
				{
					data[size++] = static_cast<limb_t>(number & limb_max());
					number >>= limb_bit_number;
				}
			}

			data.resize(size);
			negative = false;
		}
		
		// *this = *this * a + b
		void _mul_add(limb_t a, limb_t b)
		{
			limb_t carry = b;

			for (limb_t* i = data.begin(); i != data.end(); ++i)
			{
				limb2_t result = static_cast<limb2_t>(*i) * a + carry;

				*i = static_cast<limb_t>(result & limb_max());

				carry = static_cast<limb_t>(result >> limb_bit_number);
			}

			if (carry != 0) data.push_back(carry);
		}

		template <typename Ch> void _assign_str(const Ch* str, int base)
		{
			if (base < 2 && base > 36) return assign(0);
			
			// skip whitespace
			while (detail::bigint::isspace(*str)) ++str;

			negative = false;

			if (*str == Ch('-'))
			{
				negative = true;
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

			data.resize(0);
			
			for (; *str; ++str)
			{
				if (!detail::bigint::is_ascii(*str) || digit_value_tab[static_cast<unsigned int>(*str)] >= base)
				{
					break;
				}
				
				_mul_add(static_cast<limb_t>(base), digit_value_tab[static_cast<unsigned int>(*str)]);
			}
		}

		void assign(const char* str, int base)
		{
			_assign_str(str, base);
		}
		
		void assign(const wchar_t* str, int base)
		{
			_assign_str(str, base);
		}

		void _add_unsigned(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			limb_t carry = 0;
			
			size_t li_size = lhs.data.size();
			size_t ri_size = rhs.data.size();
			
			data.resize((std::max)(lhs.data.size(), rhs.data.size()) + 1);

			const limb_t* li = lhs.data.begin();
			const limb_t* li_end = li + li_size;
			const limb_t* ri = rhs.data.begin();
			const limb_t* ri_end = ri + ri_size;
			
			limb_t* i = data.begin();

			for (; li != li_end && ri != ri_end; ++li, ++ri)
			{
				limb2_t result = static_cast<limb2_t>(*li) + *ri + carry;

				*i++ = static_cast<limb_t>(result & limb_max());

				carry = static_cast<limb_t>(result >> limb_bit_number);
			}

			for (; li != li_end; ++li)
			{
				limb2_t result = static_cast<limb2_t>(*li) + carry;

				*i++ = static_cast<limb_t>(result & limb_max());

				carry = static_cast<limb_t>(result >> limb_bit_number);
			}

			for (; ri != ri_end; ++ri)
			{
				limb2_t result = static_cast<limb2_t>(*ri) + carry;

				*i++ = static_cast<limb_t>(result & limb_max());

				carry = static_cast<limb_t>(result >> limb_bit_number);
			}

			if (carry != 0)
			{
				*i = carry;
			}
			else
			{
				data.pop_back();
			}
		}

		void _normalize()
		{
			if (data.empty()) return;
			
			// strip zeroes
			const limb_t* i = data.end();

			do
			{
				--i;
			}
			while (i != data.begin() && *i == 0);

			if (i == data.begin() && *i == 0)
			{
				data.resize(0);
				negative = false;
			}
			else
			{
				data.resize((i - data.begin()) + 1);
			}
		}

		bool _sub_unsigned(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			limb_t borrow = 0;
			
			size_t li_size = lhs.data.size();
			size_t ri_size = rhs.data.size();
			
			data.resize((std::max)(lhs.data.size(), rhs.data.size()));
			
			const limb_t* li = lhs.data.begin();
			const limb_t* li_end = li + li_size;
			const limb_t* ri = rhs.data.begin();
			const limb_t* ri_end = ri + ri_size;

			limb_t* i = data.begin();

			for (; li != li_end && ri != ri_end; ++li, ++ri)
			{
				limb2_t result = static_cast<limb2_t>(*ri) + borrow;

				if (result > *li)
				{
					result = static_cast<limb2_t>(limb_max()) + 1 + *li - result;
					borrow = 1;
				}
				else
				{
					result = *li - result;
					borrow = 0;
				}

				*i++ = static_cast<limb_t>(result & limb_max());
			}

			for (; li != li_end; ++li)
			{
				limb2_t result = borrow;

				if (result > *li)
				{
					result = static_cast<limb2_t>(limb_max()) + 1 + *li - result;
					borrow = 1;
				}
				else
				{
					result = *li - result;
					borrow = 0;
				}

				*i++ = static_cast<limb_t>(result & limb_max());
			}

			for (; ri != ri_end; ++ri)
			{
				limb2_t result = static_cast<limb2_t>(*ri) + borrow;

				if (result > 0)
				{
					result = static_cast<limb2_t>(limb_max()) + 1 - result;
					borrow = 1;
				}
				else
				{
					borrow = 0;
				}

				*i++ = static_cast<limb_t>(result & limb_max());
			}

			if (borrow != 0)
			{
				// we borrowed 2^number of bits in our number - we have to subtract it
				// for this we need to complement all limbs to 2, and add 1 to the last limb.
				for (limb_t* j = data.begin(); j != data.end(); ++j)
					*j = limb_max() - *j;
			
				data[0]++;
			}
			
			_normalize();

			return borrow != 0;
		}

		void add(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			if (lhs.negative == rhs.negative) // positive + positive or negative + negative
			{
				negative = lhs.negative;
				_add_unsigned(lhs, rhs);
			}
			else if (lhs.negative) // negative + positive
			{
				negative = _sub_unsigned(rhs, lhs);
			}
			else // positive + negative
			{
				negative = _sub_unsigned(lhs, rhs);
			}
		}

		void sub(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			if (lhs.negative != rhs.negative) // positive - negative or negative - positive
			{
				negative = lhs.negative;
				_add_unsigned(lhs, rhs);
			}
			else if (lhs.negative) // negative - negative
			{
				negative = _sub_unsigned(rhs, lhs);
			}
			else // positive - positive
			{
				negative = _sub_unsigned(lhs, rhs);
			}
		}

		void _mul_unsigned_basecase(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			if (this == &lhs || this == &rhs)
			{
				bigint_default_implementation copy;
				copy.mul(lhs, rhs);
				*this = copy;
				return;
			}

			data.resize(lhs.data.size() + rhs.data.size());
			std::fill(data.begin(), data.end(), 0);

			limb_t* i = data.begin();
			
			for (const limb_t* li = lhs.data.begin(); li != lhs.data.end(); ++li, ++i)
			{
				limb_t carry = 0;

				limb_t* ci = i;

				for (const limb_t* ri = rhs.data.begin(); ri != rhs.data.end(); ++ri)
				{
					limb2_t result = static_cast<limb2_t>(*li) * *ri + *ci + carry;

					*ci++ = static_cast<limb_t>(result & limb_max());

					carry = static_cast<limb_t>(result >> limb_bit_number);
				}

				while (carry != 0)
				{
					limb2_t result = static_cast<limb2_t>(*ci) + carry;
					
					*ci++ = static_cast<limb_t>(result & limb_max());

					carry = static_cast<limb_t>(result >> limb_bit_number);
				}
			}

			_normalize();
		}

		void _mul_unsigned_fft(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			size_t lhs_size = lhs.data.size();
			size_t rhs_size = rhs.data.size();

			data.resize(lhs_size + rhs_size);

			if (&lhs != &rhs)
				bigint_fft_multiplicator<limb_bit_number>::mul(data.begin(), lhs.data.begin(), lhs_size, rhs.data.begin(), rhs_size);
			else
				bigint_fft_multiplicator<limb_bit_number>::sqr(data.begin(), lhs.data.begin(), lhs_size);

			_normalize();
		}

		void mul(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			if (lhs.is_zero() || rhs.is_zero())
			{
				assign(0);
				return;
			}

			// Some research revealed that:

			// 1. For limb counts below 512, basecase is always faster
			if (lhs.data.size() <= 512 || rhs.data.size() <= 512)
				_mul_unsigned_basecase(lhs, rhs);
			else
			{
				// if cost of 1x1 multiply is 1
				uint64_t basecase_cost = static_cast<uint64_t>(lhs.data.size()) * rhs.data.size();
				// ... the cost of 1 step of FFT (if the asymptotical performance is N*logN) is about 32

				// find FFT (uint16) size
				size_t max_size = (std::max)(lhs.data.size(), rhs.data.size());

				// round up to the nearest power of two
				size_t N = 1;
				while (N < max_size) N *= 2;

				// fix N depending on limb_type
				N = N * sizeof(limb_t) / sizeof(uint16_t);
				if (N == 0) N = 1;

				// destination size
				N *= 2;

				size_t logN = bigint_fft_multiplicator<limb_bit_number>::log2(N);

				uint64_t fft_cost = static_cast<uint64_t>(N) * logN * 32;

				if (basecase_cost < fft_cost)
					_mul_unsigned_basecase(lhs, rhs);
				else
					_mul_unsigned_fft(lhs, rhs);
			}
			
			negative = lhs.negative ? !rhs.negative : rhs.negative;
		}

		void _div_unsigned(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs, bigint_default_implementation& r)
		{
			if (rhs.data.empty())
			{
				volatile int i = 0;
				i /= i;
				return;
			}
			
			if (lhs.data.empty())
			{
				r.assign(0);
				assign(0);
				return;
			}

			if (this == &r)
			{
				bigint_default_implementation rem;
				_div_unsigned(lhs, rhs, rem);
				r = rem;
				return;
			}

			bigint_default_implementation x(lhs);
			x.negative = false;

			bigint_default_implementation y(rhs);
			y.negative = false;

			// without this our estimates for qd become very bad
			limb_t d = static_cast<limb_t>((static_cast<limb2_t>(limb_max()) + 1) / (static_cast<limb2_t>(y.data.back()) + 1));
			x._mul_add(d, 0);
			y._mul_add(d, 0);

			data.resize(x.data.size());
			r.data.resize(0);

			limb_t* p = data.end();
			limb_t* i = x.data.end();
			
			do
			{
				--i;
				--p;

				// xx = r * (limb_max() + 1) + x[i]
				bigint_default_implementation xx;
				xx.data.resize(r.data.size() + 1);
				xx.data[0] = *i;

				limb_t* xx_data = xx.data.begin() + 1;
				for (const limb_t* ri = r.data.begin(); ri != r.data.end(); ++ri)
					*xx_data++ = *ri;

				if (xx.data.size() < y.data.size())
				{
					*p = 0;
					r = xx;
				}
				else if (xx.data.size() == y.data.size())
				{
					bigint_default_implementation z;
					z.sub(xx, y);

					if (z.negative)
					{
						*p = 0;
						r = xx;
					}
					else
					{
						*p = 1;
						r = z;
					}
				}
				else
				{
					// Guess a value for q [Knuth, vol.2, section 4.3.1]
					limb_t qd;
					
					if (xx.data.back() >= y.data.back())
						qd = limb_max();
					else
						qd = static_cast<limb_t>(
							((static_cast<limb2_t>(limb_max()) + 1) * xx.data.back() + xx.data[xx.data.size()-2])
							/ y.data.back()
							);

					r = y;
					r._mul_add(qd, 0);
					r.sub(xx, r);
					
					// r = xx - qd * y
					
					if (!r.negative)
					{
						while (r.compare(y) >= 0)
						{
							++qd;
							r.sub(r, y);
						}
					}
					else
					{
						while (r.negative)
						{
							--qd;
							r.add(r, y);
						}
					}
					
					*p = static_cast<limb_t>(qd);
				}
    		}
    		while (i != x.data.begin());

    		_normalize();

    		if (!r.data.empty() && d > 1)
    		{
    			r._div_rem(d);
    		}
    		
		}

		void div(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			bool neg = lhs.negative ? !rhs.negative : rhs.negative;
			bigint_default_implementation r;
			_div_unsigned(lhs, rhs, r);
			negative = data.empty() ? false : neg;
		}

		void mod(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			bool neg = lhs.negative;
			bigint_default_implementation q;
			q._div_unsigned(lhs, rhs, *this);
			negative = data.empty() ? false : neg;
		}

		// carry = 0 or carry = 1
		template <bool complement> limb_t _convert(limb_t limb, limb_t& carry)
		{
			if (complement)
			{
				limb2_t r = static_cast<limb2_t>(limb_max() - limb) + carry;
				
				carry = static_cast<limb_t>(r >> limb_bit_number);
				return static_cast<limb_t>(r & limb_max());
			}
			else
				return limb;
		}
		
		template <bool lhs_neg, bool rhs_neg> void _or_(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			const bool neg = lhs_neg || rhs_neg; // sign bit is or-ed
			negative = neg;

			size_t li_size = lhs.data.size();
			size_t ri_size = rhs.data.size();
			
			data.resize((std::max)(lhs.data.size(), rhs.data.size()));

			const limb_t* li = lhs.data.begin();
			const limb_t* li_end = li + li_size;
			const limb_t* ri = rhs.data.begin();
			const limb_t* ri_end = ri + ri_size;
			
			limb_t* i = data.begin();

			limb_t carry = 1, lcarry = 1, rcarry = 1;
			
			for (; li != li_end && ri != ri_end; ++li, ++ri)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(*li, lcarry) | _convert<rhs_neg>(*ri, rcarry), carry);
			}

			for (; li != li_end; ++li)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(*li, lcarry) | _convert<rhs_neg>(0, rcarry), carry); // or with rhs sign bit
			}

			for (; ri != ri_end; ++ri)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(0, lcarry) | _convert<rhs_neg>(*ri, rcarry), carry); // or with lhs sign bit
			}

			_normalize();
		}

		void or_(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			if (lhs.negative)
				rhs.negative ? _or_<true, true>(lhs, rhs) : _or_<true, false>(lhs, rhs);
			else
				rhs.negative ? _or_<false, true>(lhs, rhs) : _or_<false, false>(lhs, rhs);
		}

		template <bool lhs_neg, bool rhs_neg> void _and_(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			const bool neg = lhs_neg && rhs_neg; // sign bit is and-ed
			negative = neg;

			size_t li_size = lhs.data.size();
			size_t ri_size = rhs.data.size();
			
			data.resize((std::max)(lhs.data.size(), rhs.data.size()));

			const limb_t* li = lhs.data.begin();
			const limb_t* li_end = li + li_size;
			const limb_t* ri = rhs.data.begin();
			const limb_t* ri_end = ri + ri_size;
			
			limb_t* i = data.begin();

			limb_t carry = 1, lcarry = 1, rcarry = 1;

			for (; li != li_end && ri != ri_end; ++li, ++ri)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(*li, lcarry) & _convert<rhs_neg>(*ri, rcarry), carry);
			}

			for (; li != li_end; ++li)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(*li, lcarry) & _convert<rhs_neg>(0, rcarry), carry); // and with rhs sign bit
			}

			for (; ri != ri_end; ++ri)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(0, lcarry) & _convert<rhs_neg>(*ri, rcarry), carry); // and with lhs sign bit
			}

			_normalize();
		}

		void and_(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			if (lhs.negative)
				rhs.negative ? _and_<true, true>(lhs, rhs) : _and_<true, false>(lhs, rhs);
			else
				rhs.negative ? _and_<false, true>(lhs, rhs) : _and_<false, false>(lhs, rhs);
		}

		template <bool lhs_neg, bool rhs_neg> void _xor_(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			const bool neg = lhs_neg ? !rhs_neg : rhs_neg; // sign bit is xor-ed
			negative = neg;

			size_t li_size = lhs.data.size();
			size_t ri_size = rhs.data.size();
			
			data.resize((std::max)(lhs.data.size(), rhs.data.size()));

			const limb_t* li = lhs.data.begin();
			const limb_t* li_end = li + li_size;
			const limb_t* ri = rhs.data.begin();
			const limb_t* ri_end = ri + ri_size;
			
			limb_t* i = data.begin();

			limb_t carry = 1, lcarry = 1, rcarry = 1;

			for (; li != li_end && ri != ri_end; ++li, ++ri)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(*li, lcarry) ^ _convert<rhs_neg>(*ri, rcarry), carry);
			}

			for (; li != li_end; ++li)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(*li, lcarry) ^ _convert<rhs_neg>(0, rcarry), carry); // xor with rhs sign bit
			}

			for (; ri != ri_end; ++ri)
			{
				*i++ = _convert<neg>(_convert<lhs_neg>(0, lcarry) ^ _convert<rhs_neg>(*ri, rcarry), carry); // xor with lhs sign bit
			}

			_normalize();
		}

		void xor_(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs)
		{
			if (lhs.negative)
				rhs.negative ? _xor_<true, true>(lhs, rhs) : _xor_<true, false>(lhs, rhs);
			else
				rhs.negative ? _xor_<false, true>(lhs, rhs) : _xor_<false, false>(lhs, rhs);
		}

		void not_(const bigint_default_implementation& lhs)
		{
			// ~value == -(value + 1) == -value-1
			negate(lhs);
			dec();
		}

		void negate(const bigint_default_implementation& lhs)
		{
			data = lhs.data;
			negative = !lhs.negative;
			if (data.empty()) negative = false;
		}

		void lshift(const bigint_default_implementation& lhs, boost::uint64_t rhs)
		{
			if (this == &lhs)
			{
				bigint_default_implementation copy = lhs;
				return lshift(copy, rhs);
			}

			if (lhs.data.empty())
			{
				assign(0);
				return;
			}

			if (rhs / limb_bit_number > (static_cast<boost::uint64_t>(1) << (sizeof(size_t) * 8)))
			{
				throw std::bad_alloc();
			}

			data.resize(lhs.data.size() + static_cast<size_t>(rhs / limb_bit_number));

			limb_t* di = data.begin() + rhs / limb_bit_number;

			for (limb_t* i = data.begin(); i != di; ++i)
				*i = 0;
				
			for (const limb_t* li = lhs.data.begin(); li != lhs.data.end(); ++li)
				*di++ = *li;
		
			if (rhs % limb_bit_number != 0) _mul_add(1 << (rhs % limb_bit_number), 0);

			negative = lhs.negative;
		}

		void rshift(const bigint_default_implementation& lhs, boost::uint64_t rhs)
		{
			if (this == &lhs)
			{
				bigint_default_implementation copy = lhs;
				return rshift(copy, rhs);
			}

			if (lhs.data.empty())
			{
				assign(0);
				return;
			}

			if (rhs / limb_bit_number > lhs.data.size())
			{
				assign(lhs.negative ? -1 : 0);
				return;
			}

			data.resize(lhs.data.size() - static_cast<size_t>(rhs / limb_bit_number));
				
			limb_t* di = data.begin();

			for (const limb_t* li = lhs.data.begin() + rhs / limb_bit_number; li != lhs.data.end(); ++li)
				*di++ = *li;
				
			limb_t r = (rhs % limb_bit_number != 0 ? _div_rem(1 << (rhs % limb_bit_number)) : 0);

			if (lhs.negative)
			{
				// if the result is zero, add sign bit
				if (data.empty())
				{
					assign(-1);
					return;
				}

				negative = true;

				// we need to correct the result if there was a remainder
				bool correct = (r != 0);

				if (!correct)
				{
					const limb_t* li_end = lhs.data.begin() + rhs / limb_bit_number;
					
					for (const limb_t* li = lhs.data.begin(); li != li_end; ++li)
						if (*li != 0)
						{
							correct = true;
							break;
						}
				}

				if (correct) dec();
			}
			else
			{
				negative = false;
			}
		}

		void inc()
		{
			bigint_default_implementation one;
			one.assign(1);

			add(*this, one);
		}

		void dec()
		{
			bigint_default_implementation one;
			one.assign(1);

			sub(*this, one);
		}
		
		int compare(const bigint_default_implementation& rhs) const
		{
			if (negative != rhs.negative) return negative > rhs.negative ? -1 : 1; 
			
			int result = negative ? -1 : 1;

			if (data.size() != rhs.data.size()) return result * (data.size() < rhs.data.size() ? -1 : 1);
			if (data.empty()) return 0;

			const limb_t* li = data.end();
			const limb_t* ri = rhs.data.end();

			do
			{
				--li; --ri;

				if (*li < *ri)
				{
					return -result;
				}
				else if (*li > *ri)
				{
					return result;
				}
			}
			while (li != data.begin());

			return 0;
		}

		// *this = *this / a, return division remainder
		limb_t _div_rem(limb_t a)
		{
			if (data.empty()) return 0;

			limb_t remainder = 0;

			limb_t* i = data.end();
			
			do
			{
				--i;

				limb2_t result = (static_cast<limb2_t>(remainder) << limb_bit_number) + *i;

				*i = static_cast<limb_t>(result / a);

				remainder = static_cast<limb_t>(result % a);
			}
			while (i != data.begin());

			if (data.back() == 0) data.pop_back();

			return remainder;
		}

		template <typename Ch> std::basic_string<Ch> _to_str(int base) const
		{
			if (base < 2 || base > 36) return std::basic_string<Ch>(1, Ch('0'));

			if (data.empty()) return std::basic_string<Ch>(1, Ch('0'));

			std::basic_string<Ch> result;

			bigint_default_implementation copy = *this;

			static const Ch digit_char_tab[] =
			{
				Ch('0'), Ch('1'), Ch('2'), Ch('3'), Ch('4'), Ch('5'), Ch('6'), Ch('7'), Ch('8'), Ch('9'), 
				Ch('a'), Ch('b'), Ch('c'), Ch('d'), Ch('e'), Ch('f'), Ch('g'), Ch('h'), Ch('i'), Ch('j'), 
				Ch('k'), Ch('l'), Ch('m'), Ch('n'), Ch('o'), Ch('p'), Ch('q'), Ch('r'), Ch('s'), Ch('t'), 
				Ch('u'), Ch('v'), Ch('w'), Ch('x'), Ch('y'), Ch('z')
			};

			limb_t base_power = base;
			size_t count = 1;

			while (static_cast<limb2_t>(base_power) * base <= limb_max())
			{
				base_power *= base;
				++count;
			}

			while (!copy.data.empty())
			{
				limb_t r = copy._div_rem(base_power);

				for (size_t i = 0; i < count; ++i)
				{
					result += digit_char_tab[r % base];
					r /= base;
				}
			}

			while (result.size() > 1 && result[result.size() - 1] == '0')
				result.erase(result.size() - 1);

			if (negative) result += '-';

			std::reverse(result.begin(), result.end());

			return result;
		}

		std::string str(int base) const
		{
			return _to_str<char>(base);
		}
		
		std::wstring wstr(int base) const
		{
			return _to_str<wchar_t>(base);
		}
		
		boost::uint64_t _to_uint64() const
		{
			boost::uint64_t value = 0;
			boost::uint64_t power = 1;

			for (const limb_t* i = data.begin(); i != data.end(); ++i)
			{
				value += *i * power;
				power <<= limb_bit_number;
			}
			
			return value;
		}

		template <typename T> bool can_convert_to() const
		{
			// Only integer types supported
			if (!std::numeric_limits<T>::is_integer) return false;
			
			boost::uint64_t max_value;
			
			size_t count = data.size();
			
			if (negative)
			{
				max_value = static_cast<boost::uint64_t>(-static_cast<boost::int64_t>((std::numeric_limits<T>::min)()));
			}
			else
			{
				max_value = (std::numeric_limits<T>::max)();
			}

			if (count * limb_bit_number > sizeof(boost::uint64_t) * 8) // we can't fit in uint64 => we won't fit in anything else
				return false;

			return max_value >= _to_uint64();
		}
		
		template <typename T> T to_number() const
		{
			if (!std::numeric_limits<T>::is_integer) return T();
			
			boost::uint64_t value = _to_uint64();
						
			return negative ? static_cast<T>(-static_cast<boost::int64_t>(value)) : static_cast<T>(value);
		}

		bool is_zero() const
		{
			return data.empty();
		}
		
		void abs(const bigint_default_implementation& rhs)
		{
			data = rhs.data;
			negative = false;
		}
		
		void pow(const bigint_default_implementation& lhs, boost::uint64_t rhs)
		{
			if (lhs.data.empty())
			{
				assign(rhs == 0 ? 1 : 0);
				return;
			}

			if (lhs.data.size() == 1 && lhs.data[0] == 1)
			{
				assign(lhs.negative ? (rhs % 2 ? -1 : 1) : 1);
				return;
			}

			assign(1);

			boost::uint64_t pot = 1;

			// Find largest power of two that is >= rhs
			while (pot < rhs && (pot << 1) != 0)
				pot <<= 1;
 
			// Now pot is the highest bit of rhs
			if (pot > rhs)
				pot >>= 1;

			while (pot > 0)
			{
				mul(*this, *this);
				
				if ((rhs & pot) != 0)
				{
					mul(*this, lhs);
				}
  
				pot >>= 1;
			}
		}
		
		void div(const bigint_default_implementation& lhs, const bigint_default_implementation& rhs, bigint_default_implementation& remainder)
		{
			bool q_neg = lhs.negative ? !rhs.negative : rhs.negative;
			bool r_neg = lhs.negative;
			_div_unsigned(lhs, rhs, remainder);
			negative = data.empty() ? false : q_neg;
			remainder.negative = remainder.data.empty() ? false : r_neg;
		}
		
		void sqrt(const bigint_default_implementation& lhs)
		{
			if (lhs.negative)
			{
				volatile int i = 0;
				i /= i;
				return;
			}

			if (lhs.data.empty())
			{
				assign(0);
				return;
			}

			bigint_default_implementation a; // approximation
			a.data.resize((lhs.data.size() + 1) / 2);
			
			for (limb_t* i = a.data.begin(); i != a.data.end(); ++i)
				*i = 0;
		
			a.data.back() = lhs.data.back() / 2;
			
			if (a.data.back() == 0) a.data.back() = 1;
		
			// iterate
			for (;;)
			{
				bigint_default_implementation ia;
				ia.div(lhs, a);
				
				bigint_default_implementation na;
				na.add(ia, a);
				na._div_rem(2);
				// na = (lhs / a + a) / 2

				// if |ia-a|=1, then na = min(ia, a), and it's our result
				if (na.compare(a) == 0)	// a = na
				{
					*this = na;
					return;
				}
				
				if (na.compare(ia) == 0) // a = ia
				{
					*this = na;
					return;
				}

				a = na;
			}
		}
	};
} }  // namespace boost::detail

#endif // BOOST_BIGINT_BIGINT_DEFAULT_HPP
