//  Copyright (c) 2020 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/config/defines.hpp>

#if defined(HPX_HAVE_CXX17_STD_VARIANT)
#include <hpx/type_support/pack.hpp>

#include <cstddef>    // for size_t
#include <utility>
#include <variant>

namespace hpx {

    // This is put into the same embedded namespace as the implementations in
    // tuple.hpp
    namespace adl_barrier {

        template <std::size_t I, typename... Ts>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename util::at_index<I, Ts...>::type&
            get(std::variant<Ts...>& var) noexcept
        {
            return std::get<I>(var);
        }

        template <std::size_t I, typename... Ts>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename util::at_index<I, Ts...>::type const&
            get(std::variant<Ts...> const& var) noexcept
        {
            return std::get<I>(var);
        }

        template <std::size_t I, typename... Ts>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename util::at_index<I, Ts...>::type&&
            get(std::variant<Ts...>&& var) noexcept
        {
            return std::get<I>(std::move(var));
        }

        template <std::size_t I, typename... Ts>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            typename util::at_index<I, Ts...>::type const&&
            get(std::variant<Ts...> const&& var) noexcept
        {
            return std::get<I>(std::move(var));
        }
    }    // namespace adl_barrier

    using hpx::adl_barrier::get;
}    // namespace hpx

#endif
