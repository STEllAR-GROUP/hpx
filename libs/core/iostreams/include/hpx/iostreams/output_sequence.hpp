//  Copyright (c) 2026 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
// (C) Copyright 2008 CodeRage, LLC (turkanis at coderage dot com)
// (C) Copyright 2003-2007 Jonathan Turkanis

// See http://www.boost.org/libs/iostreams for documentation.

#pragma once

#include <hpx/config.hpp>
#include <hpx/iostreams/operations_fwd.hpp>
#include <hpx/iostreams/traits.hpp>

#include <type_traits>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx::iostreams {

    namespace detail {

        HPX_CXX_CORE_EXPORT template <typename T>
        struct output_sequence_impl;
    }    // namespace detail

    HPX_CXX_CORE_EXPORT template <typename T>
    decltype(auto) output_sequence(T& t)
    {
        return detail::output_sequence_impl<T>::output_sequence(t);
    }

    namespace detail {

        //------------------Definition of output_sequence_impl------------------------//
        HPX_CXX_CORE_EXPORT template <typename T>
        struct output_sequence_impl
          : std::conditional_t<detail::is_custom_v<T>, operations<T>,
                output_sequence_impl<direct_tag>>
        {
        };

        template <>
        struct output_sequence_impl<direct_tag>
        {
            template <typename U>
            static decltype(auto) output_sequence(U& u)
            {
                return u.output_sequence();
            }
        };
    }    // namespace detail
}    // namespace hpx::iostreams

#include <hpx/config/warnings_suffix.hpp>
