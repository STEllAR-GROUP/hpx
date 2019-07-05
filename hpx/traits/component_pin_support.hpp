//  Copyright (c) 2018 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_TRAITS_COMPONENT_PIN_SUPPORT_MAY_21_2018_1246PM)
#define HPX_TRAITS_COMPONENT_PIN_SUPPORT_MAY_21_2018_1246PM

#include <hpx/config.hpp>
#include <hpx/traits/detail/wrap_int.hpp>

#include <cstdint>

namespace hpx { namespace traits
{
    ///////////////////////////////////////////////////////////////////////////
    // Customization point for component pinning
    namespace detail
    {
        struct pin_helper
        {
            template <typename Component>
            HPX_CXX14_CONSTEXPR static void call(wrap_int, Component* p)
            {
            }

            // forward the call if the component implements the function
            template <typename Component>
            HPX_CXX14_CONSTEXPR static auto call(int, Component* p)
            ->  decltype(p->pin())
            {
                p->pin();
            }
        };

        struct unpin_helper
        {
            template <typename Component>
            HPX_CONSTEXPR static bool call(wrap_int, Component* p)
            {
                return false;
            }

            // forward the call if the component implements the function
            template <typename Component>
            HPX_CONSTEXPR static auto call(int, Component* p)
            ->  decltype(p->unpin())
            {
                return p->unpin();
            }
        };

        struct pin_count_helper
        {
            template <typename Component>
            HPX_CONSTEXPR static std::uint32_t call(wrap_int, Component* p)
            {
                return 0;
            }

            // forward the call if the component implements the function
            template <typename Component>
            HPX_CONSTEXPR static auto call(int, Component* p)
            ->  decltype(p->pin_count())
            {
                return p->pin_count();
            }
        };
    }

    template <typename Component, typename Enable = void>
    struct component_pin_support
    {
        HPX_CXX14_CONSTEXPR static void pin(Component* p)
        {
            detail::pin_helper::call(0, p);
        }

        HPX_CONSTEXPR static bool unpin(Component* p)
        {
            return detail::unpin_helper::call(0, p);
        }

        HPX_CONSTEXPR static std::uint32_t pin_count(Component* p)
        {
            return detail::pin_count_helper::call(0, p);
        }
    };
}}

#endif

