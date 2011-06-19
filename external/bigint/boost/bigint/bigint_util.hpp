/* Boost bigint.hpp header file
 *
 * Copyright 2007 Arseny Kapoulkine
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_BIGINT_BIGINT_UTIL_HPP
#define BOOST_BIGINT_BIGINT_UTIL_HPP

#include <string.h>

namespace boost { namespace detail { namespace bigint {

	struct toupper
	{
		char operator()(char ch) const
		{
			return ::toupper(ch);
		}
	
		wchar_t operator()(wchar_t ch) const
		{
			return ::towupper(ch);
		}
	};

	inline bool isspace(char ch)
	{
		return ::isspace(ch) != 0;
	}
	
	inline bool isspace(wchar_t ch)
	{
		return ::iswspace(ch) != 0;
	}

	inline bool is_ascii(char ch)
	{
		return ch > 0;
	}

	inline bool is_ascii(wchar_t ch)
	{
		return ch > 0 && ch < 128;
	}

	inline size_t length(const char* str)
	{
		return ::strlen(str);
	}
	
	inline size_t length(const wchar_t* str)
	{
		return ::wcslen(str);
	}
	
	inline size_t get_bit_count(size_t count, int base)
	{
		// right now it returns more bits than needed - perhaps later this will be improved
		
		size_t bits_in_base = 0;
		
		while (base > 1)
		{
			++bits_in_base;
			base = (base + 1) / 2;
		}
		
		return count * bits_in_base;
	}
	
	template <typename T> inline std::string to_string(const T& value, int base, char)
	{
		return value.str(base);
	}

	template <typename T> inline std::wstring to_string(const T& value, int base, wchar_t)
	{
		return value.wstr(base);
	}
	
} } } // namespace boost::detail::bigint

#endif // BOOST_BIGINT_BIGINT_UTIL_HPP
