//  Copyright (c) 2014-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_CONFIG_INLINE_NAMESPACE_2014_JUL_03_0754PM)
#define HPX_CONFIG_INLINE_NAMESPACE_2014_JUL_03_0754PM

#include <hpx/config/defines.hpp>

#if defined(HPX_HAVE_CXX11_INLINE_NAMESPACES)
# define HPX_INLINE_NAMESPACE(name)  inline namespace name
#else
# define HPX_INLINE_NAMESPACE(name)  namespace name
#endif

#endif
