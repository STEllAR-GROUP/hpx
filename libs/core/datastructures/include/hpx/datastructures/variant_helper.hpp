//  Copyright (c) 2020-2023 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/datastructures/config/defines.hpp>

// we shouldn't implement hpx::get for variant if hpx::variant is an alias for
// std::variant
#if defined(HPX_DATASTRUCTURES_HAVE_ADAPT_STD_VARIANT)
#include <hpx/type_support/pack.hpp>

#include <cstddef>    // for size_t
#include <utility>
#include <variant>

namespace hpx {

    // This is put into the same embedded namespace as the implementations in
    // tuple.hpp
    namespace adl_barrier {

        template <std::size_t I, typename... Ts>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE util::at_index_t<I, Ts...>&
        get(std::variant<Ts...>& var) noexcept
        {
            return std::get<I>(var);
        }

        template <std::size_t I, typename... Ts>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            util::at_index_t<I, Ts...> const&
            get(std::variant<Ts...> const& var) noexcept
        {
            return std::get<I>(var);
        }

        template <std::size_t I, typename... Ts>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE util::at_index_t<I, Ts...>&&
        get(std::variant<Ts...>&& var) noexcept
        {
            return std::get<I>(HPX_MOVE(var));
        }

        template <std::size_t I, typename... Ts>
        constexpr HPX_HOST_DEVICE HPX_FORCEINLINE
            util::at_index_t<I, Ts...> const&&
            get(std::variant<Ts...> const&& var) noexcept
        {
            return std::get<I>(HPX_MOVE(var));
        }
    }    // namespace adl_barrier

    using namespace hpx::adl_barrier;
}    // namespace hpx

#endif
