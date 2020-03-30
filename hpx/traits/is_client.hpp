//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>

#include <cstddef>
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


