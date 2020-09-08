//  Copyright (c) 2018 Hartmut Kaiser
//
//  SPDX-License-Identifier: BSL-1.0
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#pragma once

#include <hpx/config.hpp>
#include <hpx/type_support/detail/wrap_int.hpp>

#include <cstdint>

namespace hpx { namespace traits {

    ///////////////////////////////////////////////////////////////////////////
    // Customization point for component pinning
    namespace detail {
        struct pin_helper
        {
            template <typename Component>
            static constexpr void call(wrap_int, Component*)
            {
            }

            // forward the call if the component implements the function
            template <typename Component>
            static constexpr auto call(int, Component* p) -> decltype(p->pin())
            {
                p->pin();
            }
        };

        struct unpin_helper
        {
            template <typename Component>
            static constexpr bool call(wrap_int, Component*)
            {
                return false;
            }

            // forward the call if the component implements the function
            template <typename Component>
            static constexpr auto call(int, Component* p)
                -> decltype(p->unpin())
            {
                return p->unpin();
            }
        };

        struct pin_count_helper
        {
            template <typename Component>
            static constexpr std::uint32_t call(wrap_int, Component*)
            {
                return 0;
            }

            // forward the call if the component implements the function
            template <typename Component>
            static constexpr auto call(int, Component* p)
                -> decltype(p->pin_count())
            {
                return p->pin_count();
            }
        };
    }    // namespace detail

    template <typename Component, typename Enable = void>
    struct component_pin_support
    {
        static constexpr void pin(Component* p)
        {
            detail::pin_helper::call(0, p);
        }

        static constexpr bool unpin(Component* p)
        {
            return detail::unpin_helper::call(0, p);
        }

        static constexpr std::uint32_t pin_count(Component* p)
        {
            return detail::pin_count_helper::call(0, p);
        }
    };
}}    // namespace hpx::traits
