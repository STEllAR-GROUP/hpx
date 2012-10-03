//  Copyright (c) 2012 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_CONFIG_FORCEINLINE_HPP
#define HPX_CONFIG_FORCEINLINE_HPP

#include <boost/config/suffix.hpp>

#if !defined(BOOST_FORCEINLINE)
#   if defined(_MSC_VER)
#       define BOOST_FORCEINLINE __forceinline
#   elif defined(__GNUC__) && __GNUC__ > 3
#       define BOOST_FORCEINLINE inline __attribute__ ((always_inline))
#   else
#       define BOOST_FORCEINLINE inline
#   endif
#endif

#endif
