//  Copyright (c) 2013 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_NOEXCEPT_HPP
#define HPX_CONFIG_NOEXCEPT_HPP

#include <boost/system/error_code.hpp>

#if !defined(BOOST_SYSTEM_NOEXCEPT)
#define BOOST_SYSTEM_NOEXCEPT BOOST_NOEXCEPT
#endif

#endif
