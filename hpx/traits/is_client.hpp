//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_IS_CLIENT_AUG_15_2015_0818AM)
#define HPX_TRAITS_IS_CLIENT_AUG_15_2015_0818AM

#include <hpx/config.hpp>

#include <type_traits>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    template <typename T, typename Enable = void>
    struct is_client
      : std::false_type
    {};

    template <typename T, typename Enable = void>
    struct is_client_or_client_array
      : is_client<T>
    {};

    template <typename T>
    struct is_client_or_client_array<T[]>
      : is_client<T>
    {};

    template <typename T, std::size_t N>
    struct is_client_or_client_array<T[N]>
      : is_client<T>
    {};
}}

#endif

