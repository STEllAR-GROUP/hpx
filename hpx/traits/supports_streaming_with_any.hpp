//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_SUPPORTS_STREAMING_WITH_ANY_JUL_18_2013_1005AM)
#define HPX_TRAITS_SUPPORTS_STREAMING_WITH_ANY_JUL_18_2013_1005AM

#include <hpx/config.hpp>

#include <boost/mpl/bool.hpp>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for streaming with util::any
    template <typename T, typename Enable = void>
    struct supports_streaming_with_any
      : boost::mpl::true_       // the default is to support streaming
    {};
}}

#endif

