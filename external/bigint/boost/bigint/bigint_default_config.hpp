/* Boost bigint_default_config.hpp header file
 *
 * Copyright 2007 Arseny Kapoulkine
 *
 * Distributed under the Boost Software License, Version 1.0.
 * (See accompanying file LICENSE_1_0.txt or
 * copy at http://www.boost.org/LICENSE_1_0.txt)
 */

#ifndef BOOST_BIGINT_BIGINT_DEFAULT_CONFIG_HPP
#define BOOST_BIGINT_BIGINT_DEFAULT_CONFIG_HPP

#include <boost/cstdint.hpp>

namespace boost { namespace detail {
	template <size_t N> struct bigint_default_implementation_config;

	template <> struct bigint_default_implementation_config<8>
	{
		typedef boost::uint8_t first;
		typedef boost::uint16_t second;
	};

	template <> struct bigint_default_implementation_config<16>
	{
		typedef boost::uint16_t first;
		typedef boost::uint32_t second;
	};

	template <> struct bigint_default_implementation_config<32>
	{
		typedef boost::uint32_t first;
		typedef boost::uint64_t second;
	};
} }  // namespace boost::detail

#endif // BOOST_BIGINT_BIGINT_DEFAULT_CONFIG_HPP
