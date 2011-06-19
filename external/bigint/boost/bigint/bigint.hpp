/* Boost bigint.hpp header file
 *
 * Copyright 2007 Arseny Kapoulkine
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_BIGINT_BIGINT_HPP
#define BOOST_BIGINT_BIGINT_HPP

#include <ios>
#include <string>
#include <algorithm>

#include <boost/detail/workaround.hpp>
#include <boost/cstdint.hpp>

#include <boost/bigint/bigint_config.hpp>
#include <boost/bigint/bigint_util.hpp>

namespace boost {
template <typename I> class bigint_base
{
	I impl;

public:
	bigint_base()
	{
	}
	
	bigint_base(int number)
	{
		impl.assign(number);
	}
	
	bigint_base(unsigned int number)
	{
		impl.assign(number);
	}
	
	bigint_base(int64_t number)
	{
		impl.assign(number);
	}
	
	bigint_base(uint64_t number)
	{
		impl.assign(number);
	}

	explicit bigint_base(const char* str, int base = 10)
	{
		impl.assign(str, base);
	}
	
	explicit bigint_base(const wchar_t* str, int base = 10)
	{
		impl.assign(str, base);
	}
	 
	explicit bigint_base(const std::string& str, int base = 10)
	{
		impl.assign(str.c_str(), base);
	}
	
	explicit bigint_base(const std::wstring& str, int base = 10)
	{
		impl.assign(str.c_str(), base);
	}

	// - basic arithmetic operations (addition, subtraction, multiplication, division)
	const bigint_base& operator+=(const bigint_base& other)
	{
		impl.add(impl, other.impl);
		return *this;
	}
	
	const bigint_base& operator-=(const bigint_base& other)
	{
		impl.sub(impl, other.impl);
		return *this;
	}

	const bigint_base& operator*=(const bigint_base& other)
	{
		impl.mul(impl, other.impl);
		return *this;
	}

	const bigint_base& operator/=(const bigint_base& other)
	{
		impl.div(impl, other.impl);
		return *this;
	}

	// - modulo
	const bigint_base& operator%=(const bigint_base& other)
	{
		impl.mod(impl, other.impl);
		return *this;
	}

	// - bit operations (bit logic (or, and, xor), bit shifts (left/right))
	const bigint_base& operator|=(const bigint_base& other)
	{
		impl.or_(impl, other.impl);
		return *this;
	}
	
	const bigint_base& operator&=(const bigint_base& other)
	{
		impl.and_(impl, other.impl);
		return *this;
	}
	
	const bigint_base& operator^=(const bigint_base& other)
	{
		impl.xor_(impl, other.impl);
		return *this;
	}
	
	const bigint_base& operator<<=(uint64_t other)
	{
		impl.lshift(impl, other);
		return *this;
	}
	
	const bigint_base& operator>>=(uint64_t other)
	{
		impl.rshift(impl, other);
		return *this;
	}

	const bigint_base& operator++()
	{
		impl.inc();
		return *this;
	}
	
	bigint_base operator++(int) const
	{
		bigint_base old = *this;
		impl.inc();
		return old;
	}

	const bigint_base& operator--()
	{
		impl.dec();
		return *this;
	}
	
	bigint_base operator--(int) const
	{
		bigint_base old = *this;
		impl.dec();
		return old;
	}
	
	// unary operators
	bigint_base operator+() const
	{
		return *this;
	}
	
	bigint_base operator-() const
	{
		bigint_base<I> result;
		result.impl.negate(impl);
		return result;
	}
	
	bigint_base operator~() const
	{
		bigint_base<I> result;
		result.impl.not_(impl);
		return result;
	}
	
    // implicit conversion to "bool"

#if defined(__SUNPRO_CC) && BOOST_WORKAROUND(__SUNPRO_CC, <= 0x580)

    operator bool () const
    {
        return impl;
    }

#elif defined( _MANAGED )

private:
    static void unspecified_bool( bigint_base*** )
    {
    }

    typedef void (*unspecified_bool_type)( bigint_base*** );

public:
    operator unspecified_bool_type() const // never throws
    {
        return impl.is_zero() ? 0 : unspecified_bool;
    }

#elif \
    ( defined(__MWERKS__) && BOOST_WORKAROUND(__MWERKS__, < 0x3200) ) || \
    ( defined(__GNUC__) && (__GNUC__ * 100 + __GNUC_MINOR__ < 304) )

private:
    typedef std::string (bigint_base::*unspecified_bool_type)(int) const;

public: 
    operator unspecified_bool_type() const // never throws
    {
        return impl.is_zero() ? 0 : &bigint_base::str;
    }

#else 

private:
    typedef I* bigint_base::*unspecified_bool_type;

public:
    operator unspecified_bool_type() const // never throws
    {
        return impl.is_zero() ? 0 : &bigint_base::impl;
    }

#endif

    // operator! is redundant, but some compilers need it

    bool operator! () const // never throws
    {
        return impl.is_zero();
    }

	std::string str(int base = 10) const
	{
		return impl.str(base);
	}
	
	std::wstring wstr(int base = 10) const
	{
		return impl.wstr(base);
	}

	// conversion to numeric types (including 64 bit)
	template <typename T> bool can_convert_to() const
	{
		return impl.template can_convert_to<T>();
	}
	
	template <typename T> T to_number() const
	{
		return impl.template to_number<T>();
	}

	bool operator<(const bigint_base& rhs) const
	{
		return impl.compare(rhs.impl) < 0;
	}

	bool operator<=(const bigint_base& rhs) const
	{
		return impl.compare(rhs.impl) <= 0;
	}

	bool operator>(const bigint_base& rhs) const
	{
		return impl.compare(rhs.impl) > 0;
	}
	
	bool operator>=(const bigint_base& rhs) const
	{
		return impl.compare(rhs.impl) >= 0;
	}
	
	bool operator==(const bigint_base& rhs) const
	{
		return impl.compare(rhs.impl) == 0;
	}
	
	// workaround for bigint == 0 (safe bool conversion :-/)
	bool operator==(int rhs) const
	{
		bigint_base n = rhs;
		return *this == n;
	}

	bool operator!=(const bigint_base& rhs) const
	{
		return impl.compare(rhs.impl) != 0;
	}

	// workaround for bigint != 0 (safe bool conversion :-/)
	bool operator!=(int rhs) const
	{
		bigint_base n = rhs;
		return *this != n;
	}

	// - basic arithmetic operations (addition, subtraction, multiplication, division)
	friend bigint_base operator+(const bigint_base& lhs, const bigint_base& rhs)
	{
		bigint_base<I> result;
		result.impl.add(lhs.impl, rhs.impl);
		return result;
	}

	friend bigint_base operator-(const bigint_base& lhs, const bigint_base& rhs)
	{
		bigint_base<I> result;
		result.impl.sub(lhs.impl, rhs.impl);
		return result;
	}

	friend bigint_base operator*(const bigint_base& lhs, const bigint_base& rhs)
	{
		bigint_base<I> result;
		result.impl.mul(lhs.impl, rhs.impl);
		return result;
	}

	friend bigint_base operator/(const bigint_base& lhs, const bigint_base& rhs)
	{
		bigint_base<I> result;
		result.impl.div(lhs.impl, rhs.impl);
		return result;
	}
	
	// - modulo
	friend bigint_base operator%(const bigint_base& lhs, const bigint_base& rhs)
	{
		bigint_base<I> result;
		result.impl.mod(lhs.impl, rhs.impl);
		return result;
	}

	// - bit operations (bit logic (or, and, xor), bit shifts (left/right))
	friend bigint_base operator|(const bigint_base& lhs, const bigint_base& rhs)
	{
		bigint_base<I> result;
		result.impl.or_(lhs.impl, rhs.impl);
		return result;
	}
	
	friend bigint_base operator&(const bigint_base& lhs, const bigint_base& rhs)
	{
		bigint_base<I> result;
		result.impl.and_(lhs.impl, rhs.impl);
		return result;
	}
	
	friend bigint_base operator^(const bigint_base& lhs, const bigint_base& rhs)
	{
		bigint_base<I> result;
		result.impl.xor_(lhs.impl, rhs.impl);
		return result;
	}
	
	// do we need << and >> for bigints?
	friend bigint_base operator<<(const bigint_base& lhs, boost::uint64_t rhs)
	{
		bigint_base<I> result;
		result.impl.lshift(lhs.impl, rhs);
		return result;
	}

	friend bigint_base operator>>(const bigint_base& lhs, boost::uint64_t rhs)
	{
		bigint_base<I> result;
		result.impl.rshift(lhs.impl, rhs);
		return result;
	}
	
	friend bigint_base abs(const bigint_base& value)
	{
		bigint_base result;
		result.impl.abs(value.impl);
		return result;
	}

	friend bigint_base pow(const bigint_base<I>& lhs, boost::uint64_t rhs)
	{
		bigint_base result;
		result.impl.pow(lhs.impl, rhs);
		return result;
	}

	// non-standard. Do we need to change it (introduce some equivalent to div_t type) or is it ok?
	friend bigint_base div(const bigint_base& lhs, const bigint_base& rhs, bigint_base& remainder)
	{
		bigint_base result;
		result.impl.div(lhs.impl, rhs.impl, remainder.impl);
		return result;
	}

	friend bigint_base sqrt(const bigint_base& lhs)
	{
		bigint_base result;
		result.impl.sqrt(lhs.impl);
		return result;
	}

	template <typename T, typename Tr> friend std::basic_ostream<T, Tr>& operator<<(std::basic_ostream<T, Tr>& lhs, const bigint_base& rhs)
	{
		typename std::basic_ostream<T, Tr>::sentry ok(lhs);

		if (ok)
		{
			try
			{
				std::ios_base::fmtflags flags = lhs.flags() ;
				std::ios_base::fmtflags basefield = flags & std::ios_base::basefield;
				std::ios_base::fmtflags uppercase = flags & std::ios_base::uppercase;
				std::ios_base::fmtflags showpos = flags & std::ios_base::showpos;
				std::ios_base::fmtflags showbase = flags & std::ios_base::showbase;

				int base = (basefield == std::ios_base::hex) ? 16 : (basefield == std::ios_base::oct) ? 8 : 10;

				std::basic_string<T> str = detail::bigint::to_string(rhs, base, T());

				if (uppercase && base == 16) std::transform(str.begin(), str.end(), str.begin(), detail::bigint::toupper());

				typename std::basic_string<T>::size_type pad_length = 0;

				// str[0] is safe, to_string will never return empty string
				if (showpos && str[0] != '-')
				{
					str.insert(str.begin(), '+');
					pad_length = 1;
				}
				else pad_length = (str[0] == '-');
			
				const std::numpunct<T>& punct = std::use_facet<std::numpunct<T> >(lhs.getloc());

				std::string grouping = punct.grouping();
		
				if (!grouping.empty())
				{
					std::basic_string<T> nstr;

					typename std::basic_string<T>::reverse_iterator it = str.rbegin();
					typename std::basic_string<T>::reverse_iterator end = str.rend();
					if (pad_length > 0) --end; // skip sign
		
					size_t group_id = 0;
					size_t chars_to_go = str.size() - pad_length;

					while (it != end)
					{
						char limit = group_id >= grouping.size() ? (grouping.empty() ? 0 : grouping[grouping.size() - 1]) : grouping[group_id];
				
						if (!nstr.empty()) nstr += punct.thousands_sep();

						if (limit <= 0)
						{
							nstr.append(it, end);
							break;
						}

						size_t count = (std::min)(static_cast<size_t>(limit), chars_to_go);
		
						nstr.append(it, it + count);
				
						it += count;
						chars_to_go -= count;
			
						if (group_id < grouping.size()) ++group_id;
					}
		
					std::reverse(nstr.begin(), nstr.end());

					str.replace(str.begin() + pad_length, str.end(), nstr.begin(), nstr.end());
				}
		
				if (showbase && (base == 16 || base == 8))
				{
					const T str_0X[] = {T('0'), T('X'), T()};
					const T str_0x[] = {T('0'), T('x'), T()};
					const T str_0[] = {T('0'), T()};

					str.insert(pad_length, base == 16 ? (uppercase ? str_0X : str_0x) : str_0);
					if (base == 16) pad_length += 2;
				}

				if (lhs.width() != 0)
				{
					std::streamsize width = lhs.width();
					lhs.width(0);
	
					if (width > 0 && static_cast<size_t>(width) > str.length())
					{
						std::ios_base::fmtflags adjustfield = flags & std::ios_base::adjustfield;
						
						typename std::basic_string<T>::size_type pad_pos = 0; // pad before
						
						if (adjustfield == std::ios_base::left) pad_pos = str.length();
						else if (adjustfield == std::ios_base::internal) pad_pos = pad_length;
			
						str.insert(pad_pos, width - str.length(), lhs.fill());
					}
				}
			
				return lhs << str;
			}
			catch (...)
			{
				lhs.setstate(std::ios_base::badbit); // may throw
				return lhs;
			}
		}
		else return lhs;
	}

	template <typename T, typename Tr> friend std::basic_istream<T, Tr>& operator>>(std::basic_istream<T, Tr>& lhs, bigint_base& rhs)
	{
		typename std::basic_istream<T, Tr>::sentry ok(lhs);

		if (ok)
		{
			try
			{
				std::ios_base::fmtflags flags = lhs.flags() ;
				std::ios_base::fmtflags basefield = flags & std::ios_base::basefield;

				int base = (basefield == std::ios_base::hex) ? 16 : (basefield == std::ios_base::oct) ? 8 : 10;

				int sign = 1;

				std::basic_string<T> str;

				if (flags & std::ios_base::skipws)
				{
					// skip whitespaces
					while (lhs.peek() != Tr::eof() && detail::bigint::isspace(static_cast<T>(lhs.peek())))
						lhs.get();
				}

				if (lhs.peek() == T('-') || lhs.peek() == T('+'))
				{
					sign = lhs.get() == T('-') ? -1 : 1;
				}

				T char_table[] = {T('0'), T('1'), T('2'), T('3'), T('4'), T('5'), T('6'), T('7'), T('8'), T('9'),
				                  T('a'), T('b'), T('c'), T('d'), T('e'), T('f'), T('A'), T('B'), T('C'), T('D'),
				                  T('E'), T('F'), T()};
				
				size_t char_table_size = base == 16 ? sizeof(char_table) / sizeof(char_table[0]) : base;

				if (lhs.peek() == T('0'))
				{
					lhs.get();

					if (lhs.peek() == T('x') || lhs.peek() == T('X')) // 0x 0X
						lhs.get(); // skip
					else
					{
						while (lhs.peek() == T('0')) lhs.get(); // skip zeroes

						if (Tr::find(char_table, char_table_size, lhs.peek()) == 0) // next symbol is non-digit, we needed that 0
							str += T('0');
					}
				}

				const std::numpunct<T>& punct = std::use_facet<std::numpunct<T> >(lhs.getloc());

				typename Tr::int_type ch;

				while ((ch = lhs.peek()) != Tr::eof())
				{
					// do we allow this kind of character?
					if (ch == punct.thousands_sep() || Tr::find(char_table, char_table_size, ch) != 0)
					{
						str += lhs.get();
					}
					else
					{
						// have we read any valid data?
						if (str.empty())
						{
							// no.
							lhs.setstate(std::ios_base::badbit); // may throw
							return lhs;
						}

						break;
					}
				}

				std::string grouping = punct.grouping();
		
				if (!grouping.empty())
				{
					typename std::basic_string<T>::reverse_iterator it = str.rbegin();
					typename std::basic_string<T>::reverse_iterator end = str.rend();
		
					size_t group_id = 0;

					while (it != end)
					{
						char limit = group_id >= grouping.size() ? (grouping.empty() ? 0 : grouping[grouping.size() - 1]) : grouping[group_id];
				
						typename std::basic_string<T>::reverse_iterator sep_it = std::find(it, end, punct.thousands_sep());

						if (limit <= 0) // unlimited sequence of digits
						{
							if (sep_it != str.rend()) // there's another separator, error
							{
								lhs.setstate(std::ios_base::badbit); // may throw
								return lhs;
							}

							break;
						}

						// limited sequence of digits

						// we're not at the end
						if (sep_it != str.rend())
						{
							// digit sequence sizes do not match
							if (limit != std::distance(it, sep_it))
							{
								lhs.setstate(std::ios_base::badbit); // may throw
								return lhs;
							}
							else
							{
								it = sep_it;
								++it;
							}
						}
						else if (limit < std::distance(it, sep_it)) // we're at the end, and our sequence size is larger
						{
							lhs.setstate(std::ios_base::badbit); // may throw
							return lhs;
						}
						else
						{
							// we're at the end
							break;
						}

						if (group_id < grouping.size()) ++group_id;
					}
					
					// remove all separators, we don't need them
					str.erase(std::remove(str.begin(), str.end(), punct.thousands_sep()), str.end());
				}
		
				rhs = bigint_base(str, base);
				if (sign == -1) rhs.impl.negate(rhs.impl);

				return lhs;
			}
			catch (...)
			{
				lhs.setstate(std::ios_base::badbit); // may throw
				return lhs;
			}
		}
		else return lhs;
	}
};
} // namespace boost

// Do we have GMP?
#ifdef BOOST_BIGINT_HAS_GMP_SUPPORT

#include <boost/bigint/bigint_gmp.hpp>

namespace boost {

typedef bigint_base<detail::bigint_gmp_implementation> bigint;

} // namespace boost

#else

#include <boost/bigint/bigint_default.hpp>
#include <boost/bigint/bigint_storage_vector.hpp>

namespace boost {

typedef bigint_base<detail::bigint_default_implementation<detail::bigint_storage_vector> > bigint;

} // namespace boost

#endif

#endif // BOOST_BIGINT_BIGINT_HPP
