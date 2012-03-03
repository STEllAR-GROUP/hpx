//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_FUNCTION_SEP_22_2011_1142AM)
#define HPX_CONFIG_FUNCTION_SEP_22_2011_1142AM

#include <hpx/config.hpp>

#if defined(HPX_UTIL_FUNCTION)
#  include <hpx/util/function.hpp>
#else
#if !defined(HPX_HAVE_CXX11_STD_FUNCTION)
#  include <boost/function.hpp>
#else
#  include <functional>
#endif
#endif

#endif


