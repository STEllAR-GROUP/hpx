//  Copyright (c) 2013 Agustin Berge
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_PLACEHOLDER_AUG_28_2013_0603PM)
#define HPX_TRAITS_IS_PLACEHOLDER_AUG_28_2013_0603PM

#include <hpx/config.hpp>

#include <boost/mpl/size_t.hpp>

#ifndef BOOST_NO_CXX11_HDR_FUNCTIONAL
#   include <functional>
#endif

namespace hpx { namespace traits
{
    template <typename T>
    struct is_placeholder
#   ifndef BOOST_NO_CXX11_HDR_FUNCTIONAL
      : std::is_placeholder<T>
#   else
      : boost::mpl::size_t<0>
#   endif
    {};
}}

#endif
