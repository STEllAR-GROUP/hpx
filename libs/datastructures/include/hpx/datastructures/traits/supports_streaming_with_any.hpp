//  Copyright (c) 2007-2013 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_SUPPORTS_STREAMING_WITH_ANY_JUL_18_2013_1005AM)
#define HPX_TRAITS_SUPPORTS_STREAMING_WITH_ANY_JUL_18_2013_1005AM

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx { namespace traits {
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for streaming with util::any
    template <typename T, typename Enable = void>
    struct supports_streaming_with_any
      : std::true_type    // the default is to support streaming
    {
    };
}}    // namespace hpx::traits

#endif
